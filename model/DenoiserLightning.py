import pytorch_lightning as pl
import torch
import torch.nn.functional as F


import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import wandb
import os
import pandas as pd
import os
import pickle
import re
from collections import defaultdict

from setup.make_background_subtracted_frame_to_frame_difference_frames import get_background_sub_frames
from metrics.psnr import PSNR
from utils.eval_utils import transform_bbox_annos, spatial_min_max_normalize

from losses.losses import DenoiserLoss

from dataset.Augmentations import Augmentation

class DenoiserLightning(pl.LightningModule):
    def __init__(
        self, 
        model, 
        experiment_name="exp",
        learning_rate=0.0005,
        optimizer="adamw",
        inference_clips=None,
        frames_folder=None,
        annotations_folder=None,
        target="l1",
        loss_lambda=0.5,
        in_mask_weight=0.0,
        oom_weight=5.0,
        loss="mse",
        scheduler="step",
        step_size=2,
        gamma=0.05,
        factor=0.1,
        patience=4,
        augmentations=Augmentation(),
        perc_weight=[100.0, 1.6, 2.3, 1.8, 2.8, 100.0]
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.inference_clips = inference_clips
        self.frames_folder = frames_folder
        self.annotations_folder = annotations_folder
        self.target = target
        self.global_backgrounds = {}
        self.loss_lambda = loss_lambda
        self.in_mask_weight = in_mask_weight
        self.oom_weight = oom_weight
        self.loss = loss
        self.perc_weight = perc_weight
        self.loss_fn = DenoiserLoss(loss_type=loss, loss_lambda=loss_lambda, in_mask_weight=in_mask_weight, oom_weight=oom_weight, perc_weight=perc_weight)
        self.augmentations = augmentations
        self.optimizer = optimizer
        self.current_epoch_train_losses = []
        self.current_epoch_val_losses = []
        self.train_loss = []
        self.val_loss = []
        self.experiment_name = experiment_name
        self.scheduler = scheduler
        self.step_size = step_size
        self.gamma = gamma
        self.factor = factor
        self.patience = patience
        self.save_hyperparameters(ignore=['model', 'augmentations'])
        
        if target == "bs_global":
            for clip in self.inference_clips:
                self.global_backgrounds[clip] = self.calculate_global_backgrounds(clip)

    def calculate_global_backgrounds(self, clip):
        # get all frames for a clip
        full_clip = os.path.join(self.frames_folder, clip)  
        all_frames = [f for f in os.listdir(full_clip) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]
        frames = np.stack([Image.open(os.path.join(full_clip, f)).convert('L') for f in all_frames])
        blurred_frames = frames.astype(np.float32)
        # Blur frames
        for i in range(frames.shape[0]):
            blurred_frames[i] = cv2.GaussianBlur(blurred_frames[i], (5,5), 0)
        return np.mean(blurred_frames, axis=0)


    def forward(self, x):
        return self.model(x)
    

    def training_step(self, batch, batch_idx):
        frame1, frame2, frame3, target, mask = batch
        if self.training:
            frame1, frame2, frame3, target, mask = self.augmentations(frame1, frame2, frame3, target, mask)
        x = torch.cat([frame1, frame2, frame3], dim=1)
        pred = self.forward(x)
        loss = self.loss_fn(pred, target, mask)
        self.current_epoch_train_losses.append(loss.item())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def val_step(self):
        self.model.eval()        
        with torch.no_grad():
            for clip_idx, clip_name in enumerate(self.inference_clips):
                frame_nums = [64, 200, 40, 251, 225]
                frame_num = frame_nums[clip_idx]
                
                # Get all frames and predictions
                pred, target, gt, original_shape, transform_shape, frame1, frame2, frame3, frame4, frame5, mask = \
                    self.denoise_for_validation(clip_name, frame_num)

                # Calculate validation loss
                val_loss = self.loss_fn(pred, target, mask)
                self.current_epoch_val_losses.append(val_loss.item())

                # Load bboxes and calculate PSNR
                bboxes = self.load_bboxes(clip_name, frame_num, original_shape, transform_shape)
                
                # Prepare images for PSNR
                gt_np = spatial_min_max_normalize(gt[0].cpu().numpy())
                pred_np = spatial_min_max_normalize(pred[0].cpu().numpy())

                # Calculate PSNR
                psnr = PSNR(gt_np, pred_np, bboxes, clip_name, frame_num)
                _ = psnr.ratio()

                # Log metrics
                self.log(f'snr_ratio_{clip_name}', psnr.psnr_ratio)
                # self.log(f'gt_snr_avg_{clip_name}', psnr.gt_snr_avg)
                # self.log(f'pred_snr_avg_{clip_name}', psnr.pred_snr_avg)

                # Generate and log visualization
                self.log_visualizations(
                    clip_name, 
                    pred, 
                    target, 
                    gt, 
                    psnr,
                    frame1,
                    frame2,
                    frame3,
                    frame4,
                    frame5
                )



    def on_train_epoch_end(self):
        self.val_step()
        """Custom validation with PSNR calculations"""        
        self._save_losses()
        self.current_epoch_train_losses = []
        self.current_epoch_val_losses = []

    def _save_losses(self):
        if not self.current_epoch_train_losses:
            return  # Skip if no losses to save
            
        # Calculate average train loss for current epoch
        epoch_train_loss = sum(self.current_epoch_train_losses) / len(self.current_epoch_train_losses)
        
        if len(self.current_epoch_val_losses) > 0:
            # Get current validation loss if available
            epoch_val_loss = sum(self.current_epoch_val_losses) / len(self.current_epoch_val_losses)
        else:
            epoch_val_loss = None
        
        # Update loss lists
        while len(self.train_loss) < self.current_epoch:
            self.train_loss.append(epoch_train_loss)
        if epoch_val_loss is not None:
            while len(self.val_loss) < self.current_epoch:
                self.val_loss.append(epoch_val_loss)

        # Create directory if it doesn't exist
        save_dir = os.path.join(f'checkpoints/{self.experiment_name}', 'losses')
        os.makedirs(save_dir, exist_ok=True)
        
        # Also save as text file for easy viewing
        with open(os.path.join(save_dir, 'losses.txt'), 'w') as f:
            f.write('epoch,train_loss,val_loss\n')
            for i in range(max(len(self.train_loss), len(self.val_loss))):
                train_loss = self.train_loss[i] if i < len(self.train_loss) else 'NA'
                val_loss = self.val_loss[i] if i < len(self.val_loss) else 'NA'
                f.write(f'{i+1},{train_loss},{val_loss}\n')
        
        # Create a plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_loss, label='Train Loss')
        val_epochs = list(range(1, len(self.val_loss) + 1))
        plt.plot(val_epochs, self.val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
        print(f"Loss plot saved to {os.path.join(save_dir, 'loss_plot.png')}")
        plt.close()


    def denoise_for_validation(self, clip_name, frame_num):
        """Helper method to prepare frames for validation"""
        path = f"{self.frames_folder}/{clip_name}/{frame_num}.jpg"
        
        # Load frames
        frame1 = Image.open(path).convert('L')
        original_shape = frame1.size
        frame2 = Image.open(path.replace(f'{frame_num}.jpg', f'{frame_num-1}.jpg')).convert('L')
        frame3 = Image.open(path.replace(f'{frame_num}.jpg', f'{frame_num-2}.jpg')).convert('L')
        frame4 = Image.open(path.replace(f'{frame_num}.jpg', f'{frame_num+1}.jpg')).convert('L')
        frame5 = Image.open(path.replace(f'{frame_num}.jpg', f'{frame_num+2}.jpg')).convert('L')

        # Load mask annotations if available
        if self.annotations_folder:
            anno_path = os.path.join(self.annotations_folder, clip_name, "gt.txt")
            if os.path.exists(anno_path):
                obj_mask = Image.new('L', frame1.size)
                with open(anno_path, 'r') as f:
                    all_annos = f.readlines()
                # get annotations for frame4
                frame_id = int(re.findall(r'(\d+)\.jpg', path)[0])
                annos = [anno for anno in all_annos if int(anno.split(',')[0]) == frame_id+1] # +1 because MOT is 1-indexed
                # mot annos are frame_id, track_id, x, y, w, h, conf, x1, y1, z1
                for anno in annos:
                    _, _, x, y, w, h, conf, _, _, _ = map(int, map(float, anno.split(',')))
                    obj_mask.paste(1, (x, y, x+w, y+h))
            else:
                # make the obj mask all 1s so no penalty is applied
                obj_mask = Image.new('L', frame1.size)
                obj_mask.paste(1, (0,0, obj_mask.width, obj_mask.height))
        else:
            # make the obj mask all 1s so no penalty is applied
            obj_mask = Image.new('L', frame1.size)
            obj_mask.paste(1, (0,0, obj_mask.width, obj_mask.height))

        # Transform frames
        transform = self.trainer.datamodule.transform
        frame1 = transform(frame1).to(self.device)
        transform_shape = frame1.shape
        frame2 = transform(frame2).to(self.device)
        frame3 = transform(frame3).to(self.device)
        frame4 = transform(frame4).to(self.device)
        frame5 = transform(frame5).to(self.device)
        obj_mask = transform(obj_mask).to(self.device)

        # Calculate target based on target type
        if self.target == "l1":
            target = torch.abs(frame4 - frame5)
        elif self.target == "sum_minus_3_background":
            # Load global background if not already loaded
            if not hasattr(self, 'global_backgrounds'):
                self.global_backgrounds = {}
            if clip_name not in self.global_backgrounds:
                self.global_backgrounds[clip_name] = self.calculate_global_backgrounds(clip_name)
            background = Image.fromarray(self.global_backgrounds[clip_name].astype(np.uint8))
            background = transform(background).to(self.device)
            target_non_abs = frame1 + frame2 + frame3 - 3*background
            target_positive = torch.where(target_non_abs > 0, target_non_abs, torch.zeros_like(target_non_abs))
            target = target_positive
        elif self.target == "sum_minus_5_background":
            # Load global background if not already loaded
            if not hasattr(self, 'global_backgrounds'):
                self.global_backgrounds = {}
            if clip_name not in self.global_backgrounds:
                self.global_backgrounds[clip_name] = self.calculate_global_backgrounds(clip_name)
            background = Image.fromarray(self.global_backgrounds[clip_name].astype(np.uint8))
            background = transform(background).to(self.device)
            target_non_abs = frame1 + frame2 + frame3 + frame4 + frame5 - 5*background
            target_positive = torch.where(target_non_abs > 0, target_non_abs, torch.zeros_like(target_non_abs))
            target = target_positive
        elif self.target == "gap2_and_curr_frame":
            future_diff = frame5 - frame1 # exaggerates the motion in the forward direction
            future_diff = torch.where(future_diff > 0, future_diff, torch.zeros_like(future_diff))
            prev_diff = frame3 - frame1 # exaggerates the motion in the backward direction
            prev_diff = torch.where(prev_diff > 0, prev_diff, torch.zeros_like(prev_diff))
            target = frame1 + future_diff + prev_diff
        elif self.target == "gap1_and_curr_frame":
            future_diff = frame4 - frame1 # exaggerates the motion in the forward direction
            future_diff = torch.where(future_diff > 0, future_diff, torch.zeros_like(future_diff))
            prev_diff = frame2 - frame1 # exaggerates the motion in the backward direction
            prev_diff = torch.where(prev_diff > 0, prev_diff, torch.zeros_like(prev_diff))
            target = frame1 + future_diff + prev_diff
        elif self.target == "gap1_inverted_and_curr_frame":
            future_diff = frame4 - frame1 # exaggerates the motion in the forward direction
            future_diff = torch.where(future_diff <= 0, future_diff, torch.zeros_like(future_diff))
            prev_diff = frame2 - frame1 # exaggerates the motion in the backward direction
            prev_diff = torch.where(prev_diff <= 0, prev_diff, torch.zeros_like(prev_diff))
            target = frame1 + future_diff + prev_diff
        elif self.target == "all_frame_diffs_and_curr":
            target = torch.abs(frame5 - frame4) + torch.abs(frame4 - frame3) + torch.abs(frame3 - frame2) + torch.abs(frame2 - frame1) + frame1
        elif self.target == "positive_diff_and_curr_frame":
            future_diff = frame5 - frame4 # exaggerates the motion in the forward direction
            future_diff = torch.where(future_diff > 0, future_diff, torch.zeros_like(future_diff))
            prev_diff = frame2 - frame1 # exaggerates the motion in the backward direction
            prev_diff = torch.where(prev_diff > 0, prev_diff, torch.zeros_like(prev_diff))
            target = frame1 + future_diff + prev_diff
        elif self.target == "identity":
            target = frame1
        elif self.target == "identity_t-1":
            target = frame2
        elif self.target == "l1_and_obj":
            target = torch.abs(frame4 - frame5)
        elif self.target == "bs_local":
            # Convert frames to numpy for background subtraction
            frames_np = np.stack([
                frame3.cpu().numpy(),
                frame2.cpu().numpy(),
                frame1.cpu().numpy(),
                frame4.cpu().numpy(),
                frame5.cpu().numpy()
            ])
            bs_frames = get_background_sub_frames(frames_np)
            target = torch.tensor(bs_frames[2]).to(self.device)
        elif self.target == "bs_global":
            # Load global background if not already loaded
            if not hasattr(self, 'global_backgrounds'):
                self.global_backgrounds = {}
            
            if clip_name not in self.global_backgrounds:
                self.global_backgrounds[clip_name] = self.calculate_global_backgrounds(clip_name)
            
            background = Image.fromarray(self.global_backgrounds[clip_name].astype(np.uint8))
            background = transform(background).to(self.device)
            target = frame1 - background
        elif self.target == "stddev_all":
            target = torch.std(torch.stack([frame3, frame2, frame1, frame4, frame5]), dim=0)
        elif self.target == "stddev_3":
            target = torch.std(torch.stack([frame2, frame1, frame4]), dim=0)
        else:
            raise ValueError(f"Unknown target type: {self.target}")

        # Verify target shape
        assert target.shape == frame1.shape, f"Target shape {target.shape} doesn't match input shape {frame1.shape}"
       
        target = target.to(self.device)

        # Get prediction
        frames = [frame1, frame2, frame3]
        frames = [f.unsqueeze(0) if f.dim() == 3 else f for f in frames]
        x = torch.cat(frames, dim=1)
        pred = self(x)

        pred = pred.squeeze(0)
        return pred, target, frame4, original_shape, transform_shape, frame1, frame2, frame3, frame4, frame5, obj_mask


    def load_bboxes(self, clip_name, frame_num, original_shape, transform_shape):
        """Helper method to load and transform bboxes"""
        mot_path = f"{self.annotations_folder}/{clip_name}/gt.txt"
        mot_annotations = pd.read_csv(
            mot_path, 
            delimiter=',', 
            header=None,
            names=['frame', 'id', 'x1', 'y1', 'w', 'h', 'conf', 'x_off', 'y_off', 'z_off']
        )
        mot_annotations_frame = mot_annotations[mot_annotations['frame'] == frame_num+1]
        mot_annotations_frame = transform_bbox_annos(
            mot_annotations_frame, 
            original_shape, 
            transform_shape
        )
        return mot_annotations_frame[['x1', 'y1', 'w', 'h']].values.tolist()

    def log_visualizations(self, clip_name, pred, target, gt, psnr, frame1, frame2, frame3, frame4, frame5):
        """Helper method to create and log visualizations with all frames
        Args:
            clip_name: Name of the clip
            pred: Predicted output
            target: Target output
            gt: Ground truth frame
            psnr: PSNR object
            frame1-5: Input frames in sequence
        """
        # Create visualization figure with all frames and outputs
        fig, ax = plt.subplots(2, 5, figsize=(15, 10))
        # First row: Input sequence
        ax[0, 0].imshow(frame3[0].cpu().numpy(), cmap='gray')
        ax[0, 0].set_title('Frame 3 (Previous-2)')
        ax[0, 1].imshow(frame2[0].cpu().numpy(), cmap='gray')
        ax[0, 1].set_title('Frame 2 (Previous)')
        ax[0, 2].imshow(frame1[0].cpu().numpy(), cmap='gray')
        ax[0, 2].set_title('Frame 1 (Current)')
        ax[0, 3].imshow(frame4[0].cpu().numpy(), cmap='gray')
        ax[0, 3].set_title('Frame 4 (Next)')
        ax[0, 4].imshow(frame5[0].cpu().numpy(), cmap='gray')
        ax[0, 4].set_title('Frame 5 (Next+1)')
        
        # Second row: Prediction, Target, Ground Truth
        ax[1, 0].imshow(pred[0].cpu().numpy(), cmap='gray')
        ax[1, 0].set_title(f'Prediction')
        ax[1, 1].imshow(target[0].cpu().numpy(), cmap='gray')
        ax[1, 1].set_title(f'Target')
        ax[1, 2].imshow(gt[0].cpu().numpy(), cmap='gray')
        ax[1, 2].set_title(f'Ground Truth')
        
        # Add PSNR metrics to the plot
        metrics_text = f'GT SNR: {psnr.gt_snr_avg:.2f}\nPred SNR: {psnr.pred_snr_avg:.2f}\nRatio: {psnr.psnr_ratio:.2f}'
        ax[1, 3].text(0.1, 0.5, metrics_text, fontsize=14)
        ax[1, 3].axis('off')
        
        # Keep the last subplot empty or use it for additional metrics
        ax[1, 4].axis('off')
        
        # Adjust layout and add title
        plt.suptitle(f'Validation Results - {clip_name}', fontsize=16)
        plt.tight_layout()

        # Create save directory if it doesn't exist
        save_dir = os.path.join(self.trainer.checkpoint_callbacks[0].dirpath, 'visualizations')
        os.makedirs(save_dir, exist_ok=True)

        # Save
        fig.savefig(os.path.join(save_dir, f'validation_viz_{clip_name}_epoch_{self.current_epoch}.png'))
        psnr_fig = psnr.plot_comparison()
        psnr_fig.savefig(os.path.join(save_dir, f'psnr_plot_{clip_name}_epoch_{self.current_epoch}.png'))
        
        # Log to wandb if wandb exists
        if self.logger is not None:
            if isinstance(self.logger, pl.loggers.WandbLogger):
                self.logger.experiment.log({
                    f'validation_viz_{clip_name}': wandb.Image(fig),
                    f'psnr_plot_{clip_name}': wandb.Image(psnr_fig)
                })
            # otherwise just rely on the local save

        plt.close(fig)
        plt.close(psnr_fig)


    def configure_optimizers(self):
        if self.optimizer == "adamw":
            print(f"Using AdamW optimizer with learning rate {self.learning_rate}")
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "sgd":
            print(f"Using SGD optimizer with learning rate {self.learning_rate}")
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "rmsprop":
            print(f"Using RMSprop optimizer with learning rate {self.learning_rate}")
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        
        if self.scheduler == "step":
            print(f"Using StepLR scheduler with step size {self.step_size} and gamma {self.gamma}")
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        elif self.scheduler == "plateau":
            print(f"Using ReduceLROnPlateau scheduler with factor {self.factor} and patience {self.patience}")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.factor, patience=self.patience, verbose=True)
        # elif self.scheduler == "manual":
        #     print(f"Using manual scheduler ")
        #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler}")
        # try to alter the optimizer and the scheduler
        # SGD, RMSProp, Adam (noW) w/lower LR
        # try to vary learning rate, step size, gamma by 1/2 or 2x
        # scheduler reduce_on_plateau for i.e. x epochs
        # Can look into cosine annealing (but usually used for transformers)
        # manual: constant LR and then define a manual scheduler (i.e. 10 epochs at current LR, 5 epochs at 1/2 current LR, 5 at 1/20 current LR)
        # could try adding dropout to some of the layers -- 0.05, 0.1, 0.2, 0.3
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
            },
        }