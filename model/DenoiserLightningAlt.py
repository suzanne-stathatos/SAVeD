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
from einops import rearrange
from setup.make_background_subtracted_frame_to_frame_difference_frames import get_background_sub_frames
from metrics.psnr import PSNR
from utils.eval_utils import transform_bbox_annos, spatial_min_max_normalize

from losses.losses import DenoiserLoss

from dataset.Augmentations import Augmentation

class DenoiserLightningAlt(pl.LightningModule):
    def __init__(
        self, 
        model, 
        learning_rate=0.0005,
        optimizer='adam',
        inference_clips=None,
        frames_folder=None,
        annotations_folder=None,
        target="l1",
        loss_lambda=0.5,
        in_mask_weight=0.0,
        oom_weight=5.0,
        loss="mse",
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
        frames, curr_idx, target = batch
        # disable augmentations for now
        # if self.training:
        #     frames, curr_idx, target = self.augmentations(frames, curr_idx, target, mask)
        frames = frames[:,:1] # make single channel
        pred = self.forward(frames)
        pred = pred.squeeze(-1)
        loss = self.loss_fn(pred, target)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss


    def on_train_epoch_end(self):
        """Custom validation with PSNR calculations"""
        # Get current training loss
        train_loss = self.trainer.callback_metrics['train_loss']
        
        # Initialize loss lists if not exists
        if not hasattr(self, 'train_losses'):
            self.train_losses = []
        if not hasattr(self, 'val_losses'):
            self.val_losses = defaultdict(list)
        
        # Append training loss
        self.train_losses.append(train_loss.item())
        
        self.model.eval()
        with torch.no_grad():
            epoch_val_losses = {}
            for clip_idx, clip_name in enumerate(self.inference_clips):
                frame_nums = [64, 200, 40, 251, 225]
                frame_num = frame_nums[clip_idx]
                
                # Get all frames and predictions
                pred, target, frames = self.denoise_for_validation(clip_name, frame_num)

                # Calculate validation loss
                val_loss = self.loss_fn(pred, target)
                self.log(f'val_loss_{clip_name}', val_loss)
                epoch_val_losses[clip_name] = val_loss.item()
            
            # Log validation losses
            for clip_name, loss in epoch_val_losses.items():
                self.log(f'val_loss_{clip_name}', loss)
                if clip_name not in self.val_losses:
                    self.val_losses[clip_name] = []
                self.val_losses[clip_name].append(loss)

                # Save losses to pickle files
                save_dir = os.path.join(self.trainer.checkpoint_callbacks[0].dirpath, 'losses')
                os.makedirs(save_dir, exist_ok=True)
                with open(os.path.join(save_dir, 'train_losses.pkl'), 'wb') as f:
                    pickle.dump(self.train_losses, f)
                with open(os.path.join(save_dir, 'val_losses.pkl'), 'wb') as f:
                    pickle.dump(self.val_losses, f)

                # # Load bboxes and calculate PSNR
                # bboxes = self.load_bboxes(clip_name, frame_num, original_shape, transform_shape)
                
                # # Prepare images for PSNR
                # gt_np = spatial_min_max_normalize(gt[0].cpu().numpy())
                # pred_np = spatial_min_max_normalize(pred[0].cpu().numpy())

                # Calculate PSNR
                # psnr = PSNR(gt_np, pred_np, bboxes, clip_name, frame_num)
                # _ = psnr.ratio()

                # Log metrics
                # self.log(f'snr_ratio_{clip_name}', psnr.psnr_ratio)
                # self.log(f'gt_snr_avg_{clip_name}', psnr.gt_snr_avg)
                # self.log(f'pred_snr_avg_{clip_name}', psnr.pred_snr_avg)

                # Generate and log visualization
                self.log_visualizations(
                    clip_name, 
                    pred, 
                    target, 
                    frames
                )

    def denoise_for_validation(self, clip_name, frame_num):
        """Helper method to prepare frames for validation"""
        path = f"{self.frames_folder}/{clip_name}/{frame_num}.jpg"
        # Load frames
        frames = []
        for i in range(frame_num-31//2, frame_num+31//2+1):
            frames.append(Image.open(path.replace(f'{frame_num}.jpg', f'{i}.jpg')).convert('L'))

        frame1 = Image.open(path).convert('L')
        original_shape = frame1.size
        
        # Transform frames
        transform = self.trainer.datamodule.transform
        frame1 = transform(frame1).to(self.device)
        transform_shape = frame1.shape
        frames = [transform(frame).to(self.device) for frame in frames]

        # Calculate target based on target type
        curr_idx = 31//2
        if self.target == "l1":
            target = torch.abs(frames[curr_idx] - frames[curr_idx+1])
        elif self.target == "sum_minus_3_background":
            # Load global background if not already loaded
            if not hasattr(self, 'global_backgrounds'):
                self.global_backgrounds = {}
            if clip_name not in self.global_backgrounds:
                self.global_backgrounds[clip_name] = self.calculate_global_backgrounds(clip_name)
            background = Image.fromarray(self.global_backgrounds[clip_name].astype(np.uint8))
            background = transform(background).to(self.device)
            target_non_abs = frames[curr_idx] + frames[curr_idx-1] + frames[curr_idx+1] - 3*background
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
            target_non_abs = frames[curr_idx] + frames[curr_idx-1] + frames[curr_idx+1] + frames[curr_idx+2] + frames[curr_idx-2] - 5*background
            target_positive = torch.where(target_non_abs > 0, target_non_abs, torch.zeros_like(target_non_abs))
            target = target_positive
        elif self.target == "gap2_and_curr_frame":
            future_diff = frames[curr_idx+2] - frames[curr_idx] # exaggerates the motion in the forward direction
            future_diff = torch.where(future_diff > 0, future_diff, torch.zeros_like(future_diff))
            prev_diff = frames[curr_idx-2] - frames[curr_idx] # exaggerates the motion in the backward direction
            prev_diff = torch.where(prev_diff > 0, prev_diff, torch.zeros_like(prev_diff))
            target = frames[curr_idx] + future_diff + prev_diff
        elif self.target == "gap1_and_curr_frame":
            future_diff = frames[curr_idx+1] - frames[curr_idx] 
            future_diff = torch.where(future_diff > 0, future_diff, torch.zeros_like(future_diff))
            prev_diff = frames[curr_idx-1] - frames[curr_idx] # exaggerates the motion in the backward direction
            prev_diff = torch.where(prev_diff > 0, prev_diff, torch.zeros_like(prev_diff))
            target = frames[curr_idx] + future_diff + prev_diff
        elif self.target == "all_frame_diffs_and_curr":
            target = sum([torch.abs(frames[curr_idx+i] - frames[curr_idx+i-1]) for i in range(1, self.window_size//2)]) + frames[curr_idx]
        elif self.target == "positive_diff_and_curr_frame":
            future_diff = frames[curr_idx+2] - frames[curr_idx+1] # exaggerates the motion in the forward direction
            future_diff = torch.where(future_diff > 0, future_diff, torch.zeros_like(future_diff))
            prev_diff = frames[curr_idx-1] - frames[curr_idx] # exaggerates the motion in the backward direction
            prev_diff = torch.where(prev_diff > 0, prev_diff, torch.zeros_like(prev_diff))
            target = frames[curr_idx] + future_diff + prev_diff
        elif self.target == "identity":
            target = frames[curr_idx]
        elif self.target == "identity_t-1":
            target = frames[curr_idx-1]
        elif self.target == "l1_and_obj":
            target = torch.abs(frames[curr_idx+1] - frames[curr_idx+2])
        elif self.target == "l2":
            target = (frames[curr_idx+1] - frames[curr_idx+2]) ** 2
        elif self.target == "bs_local":
            # Convert frames to numpy for background subtraction
            frames_np = np.stack([frame.cpu().numpy() for frame in frames])
            bs_frames = get_background_sub_frames(frames_np)
            target = torch.tensor(bs_frames[curr_idx]).to(self.device)
        elif self.target == "bs_global":
            # Load global background if not already loaded
            if not hasattr(self, 'global_backgrounds'):
                self.global_backgrounds = {}
            if clip_name not in self.global_backgrounds:
                self.global_backgrounds[clip_name] = self.calculate_global_backgrounds(clip_name)
            
            background = Image.fromarray(self.global_backgrounds[clip_name].astype(np.uint8))
            background = transform(background).to(self.device)
            target = frames[curr_idx] - background
        elif self.target == "stddev_all":
            target = torch.std(frames, dim=0)
        elif self.target == "stddev_3":
            target = torch.std(torch.stack([frames[curr_idx-1], frames[curr_idx], frames[curr_idx+1]]), dim=0)
        else:
            raise ValueError(f"Unknown target type: {self.target}")

        # Verify target shape
        assert target.shape == frame1.shape, f"Target shape {target.shape} doesn't match input shape {frame1.shape}"
       
        target = target.to(self.device)
        frames = torch.stack(frames)
        frames -= torch.min(frames)
        frames /= torch.max(frames)

        target -= torch.min(target)
        target /= torch.max(target)

        frames = rearrange(frames, 't c h w -> c h w t')
        # include batch dimension
        frames = frames.unsqueeze(0)
        target = target.unsqueeze(0)

        # make single channel
        # if num_channels == 1:
        frames = frames[:,:1]
        # Get prediction
        pred = self(frames)
        pred = pred.squeeze(-1)
        pred = pred.squeeze(0)
        frames = frames.squeeze(0)
        target = target.squeeze(0)
        return pred, target, frames

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

    def log_visualizations(self, clip_name, pred, target, frames):
        """Helper method to create and log visualizations with all frames
        Args:
            clip_name: Name of the clip
            pred: Predicted output
            target: Target output
            frames: Input frames in sequence
        """
        # Create visualization figure with all frames and outputs
        # convert frames from c h w t to t c h w
        frames = rearrange(frames, 'c h w t -> t h w c')
        num_frames = frames.shape[0]

        # Create visualization figure with all frames, prediction, and target
        fig, ax = plt.subplots(2, num_frames, figsize=(3 * num_frames, 10))
        
        # First row: Input sequence
        for i in range(num_frames):
            ax[0, i].imshow(frames[i].cpu().numpy(), cmap='gray')
            ax[0, i].set_title(f'Frame {i+1}')
        
        # Second row: Prediction and Target
        ax[1, 0].imshow(pred[0].cpu().numpy(), cmap='gray')
        ax[1, 0].set_title('Prediction')
        ax[1, 1].imshow(target[0].cpu().numpy(), cmap='gray')
        ax[1, 1].set_title('Target')
        
        # Hide unused subplots in the second row if there are more frames than prediction/target
        for i in range(2, num_frames):
            ax[1, i].axis('off')
        
        # Adjust layout and add title
        plt.suptitle(f'Validation Results - {clip_name}', fontsize=16)
        plt.tight_layout()

        # Create save directory if it doesn't exist
        save_dir = os.path.join(self.trainer.checkpoint_callbacks[0].dirpath, 'visualizations')
        os.makedirs(save_dir, exist_ok=True)
        # Save
        fig.savefig(os.path.join(save_dir, f'validation_viz_{clip_name}_epoch_{self.current_epoch}.png'))

        # also plot train losses in another axis
        plt.figure()
        train_loss = self.train_losses
        plt.plot(train_loss)
        plt.title('Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(save_dir, f'train_loss.png'))

        # psnr_fig = psnr.plot_comparison()
        # psnr_fig.savefig(os.path.join(save_dir, f'psnr_plot_{clip_name}_epoch_{self.current_epoch}.png'))
        
        # Log to wandb if wandb exists
        if self.logger is not None:
            if isinstance(self.logger, pl.loggers.WandbLogger):
                self.logger.experiment.log({
                    f'validation_viz_{clip_name}': wandb.Image(fig),
                    # f'psnr_plot_{clip_name}': wandb.Image(psnr_fig)
                })
            # otherwise just rely on the local save

        plt.close(fig)
        # plt.close(psnr_fig)


    def configure_optimizers(self):
        if self.optimizer == 'adamw' or self.optimizer == 'adam':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.05)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
            },
        }