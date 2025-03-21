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
import re
from setup.make_background_subtracted_frame_to_frame_difference_frames import get_background_sub_frames
from metrics.psnr import PSNR
from utils.eval_utils import transform_bbox_annos, spatial_min_max_normalize
from losses.losses import weighted_bce_loss

class DenoiserLightning_prev_frames_5(pl.LightningModule):
    def __init__(
        self, 
        model, 
        learning_rate=0.0005,
        inference_clips=None,
        frames_folder=None,
        annotations_folder=None,
        target="l1"
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.inference_clips = inference_clips
        self.frames_folder = frames_folder
        self.annotations_folder = annotations_folder
        self.target = target
        self.global_backgrounds = {}
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
        for i in range(frames.shape[0]):
            blurred_frames[i] = cv2.GaussianBlur(blurred_frames[i], (5,5), 0)
        return np.mean(blurred_frames, axis=0)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        frame1, frame2, frame3, frame4, frame5, frame6, target, mask = batch
        x = torch.cat([frame1, frame2, frame3, frame4, frame5, frame6], dim=1)
        pred = self.forward(x)
        if self.target == "l1_and_obj":
            l1_loss = F.mse_loss(pred, target)
            # assert that target is normalized 0-1
            assert torch.all(target >= 0) and torch.all(target <= 1), "Target is not normalized 0-1"
            assert torch.all(mask >= 0) and torch.all(mask <= 1), "Mask is not normalized 0-1"
            obj_pixels = mask.sum()
            if obj_pixels > 0:
                # use sigmoid activation function on pred
                pred = torch.sigmoid(pred)
                assert torch.all(pred >= 0) and torch.all(pred <= 1), "Pred is not normalized 0-1"
                # do not penalize any pixels in the mask
                obj_loss = torch.tensor(weighted_bce_loss ( pred, mask, in_mask_weight=self.in_mask_weight, oom_weight=self.oom_weight), device=pred.device)
            else:
                obj_loss = torch.tensor(0.0, device=pred.device)
            total_loss = l1_loss * self.loss_lambda + obj_loss * (1 - self.loss_lambda)
            loss = total_loss
        else:
            loss = F.mse_loss(pred, target)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss


    def on_train_epoch_end(self):
        """Custom validation with PSNR calculations"""
        self.model.eval()
        with torch.no_grad():
            for clip_idx, clip_name in enumerate(self.inference_clips):
                frame_nums = [64, 200, 40, 251]
                frame_num = frame_nums[clip_idx]
                
                pred, target, gt, original_shape, transform_shape, frames, mask = \
                    self.denoise_for_validation(clip_name, frame_num)

                if self.target == "l1_and_obj":
                    l1_loss = F.mse_loss(pred, target)
                    # assert that target is normalized 0-1
                    assert torch.all(target >= 0) and torch.all(target <= 1), "Target is not normalized 0-1"
                    assert torch.all(mask >= 0) and torch.all(mask <= 1), "Mask is not normalized 0-1"
                    obj_pixels = mask.sum()
                    if obj_pixels > 0:
                        # use sigmoid activation function on pred
                        pred = torch.sigmoid(pred)
                        assert torch.all(pred >= 0) and torch.all(pred <= 1), "Pred is not normalized 0-1"
                        # do not penalize any pixels in the mask
                        obj_loss = torch.tensor(weighted_bce_loss ( pred, mask, in_mask_weight=self.in_mask_weight, oom_weight=self.oom_weight), device=pred.device)
                    else:
                        obj_loss = torch.tensor(0.0, device=pred.device)
                    total_loss = l1_loss * self.loss_lambda + obj_loss * (1 - self.loss_lambda)
                    val_loss = total_loss
                else:
                    val_loss = F.mse_loss(pred, target)
                
                self.log(f'val_loss_{clip_name}', val_loss)

                bboxes = self.load_bboxes(clip_name, frame_num, original_shape, transform_shape)
                
                gt_np = spatial_min_max_normalize(gt[0].cpu().numpy())
                pred_np = spatial_min_max_normalize(pred[0].cpu().numpy())

                psnr = PSNR(gt_np, pred_np, bboxes, clip_name, frame_num)
                _ = psnr.ratio()

                self.log(f'snr_ratio_{clip_name}', psnr.psnr_ratio)
                self.log(f'gt_snr_avg_{clip_name}', psnr.gt_snr_avg)
                self.log(f'pred_snr_avg_{clip_name}', psnr.pred_snr_avg)

                self.log_visualizations(clip_name, pred, target, gt, psnr, *frames)


    def denoise_for_validation(self, clip_name, frame_num):
        """Helper method to prepare frames for validation"""
        path = f"{self.frames_folder}/{clip_name}/{frame_num}.jpg"
        
        # Load frames (6 previous frames + 2 future frames for target calculation)
        frames = []
        for i in range(-5, 3):  # -5 to 2
            frame_path = path.replace(f'{frame_num}.jpg', f'{frame_num+i}.jpg')
            frame = Image.open(frame_path).convert('L')
            if i == -5:  # First frame
                original_shape = frame.size
            frames.append(frame)

        # Load mask annotations if available
        if self.annotations_folder:
            anno_path = os.path.join(self.annotations_folder, clip_name, "gt.txt")
            if os.path.exists(anno_path):
                obj_mask = Image.new('L', frames[0].size)
                with open(anno_path, 'r') as f:
                    all_annos = f.readlines()
                # get annotations for current img_path
                frame_id = int(re.findall(r'(\d+)\.jpg', path)[0])
                annos = [anno for anno in all_annos if int(anno.split(',')[0]) == frame_id+1] # +1 because MOT is 1-indexed
                # mot annos are frame_id, track_id, x, y, w, h, conf, x1, y1, z1
                for anno in annos:
                    _, _, x, y, w, h, conf, _, _, _ = map(int, map(float, anno.split(',')))
                    obj_mask.paste(1, (x, y, x+w, y+h))
            else:
                # make the obj mask all 1s so no penalty is applied
                obj_mask = Image.new('L', frames[0].size)
                obj_mask.paste(1, (0,0, obj_mask.width, obj_mask.height))
        else:
            # make the obj mask all 1s so no penalty is applied
            obj_mask = Image.new('L', frames[0].size)
            obj_mask.paste(1, (0,0, obj_mask.width, obj_mask.height))


        # Transform frames
        transform = self.trainer.datamodule.transform
        transformed_frames = []
        for i, frame in enumerate(frames):
            t_frame = transform(frame).to(self.device)
            if i == 0:  # First frame
                transform_shape = t_frame.shape
            transformed_frames.append(t_frame)

        # Calculate target based on target type
        if self.target == "l1":
            target = torch.abs(transformed_frames[6] - transformed_frames[7])  # Next - Next+1
        elif self.target == "l1_and_obj":
            target = torch.abs(transformed_frames[6] - transformed_frames[7])  # Next - Next+1
        elif self.target == "bs_local":
            frames_np = np.stack([f.cpu().numpy() for f in transformed_frames])
            bs_frames = get_background_sub_frames(frames_np)
            target = torch.tensor(bs_frames[5]).to(self.device)  # Current frame background subtraction
        elif self.target == "bs_global":
            background = Image.fromarray(self.global_backgrounds[clip_name].astype(np.uint8))
            background = transform(background).to(self.device)
            target = transformed_frames[5] - background  # Current frame - global background
        elif self.target == "stddev_all":
            target = torch.std(torch.stack(transformed_frames), dim=0)
        elif self.target == "stddev_3":
            target = torch.std(torch.stack([transformed_frames[4], transformed_frames[5], transformed_frames[6]]), dim=0)
        else:
            raise ValueError(f"Unknown target type: {self.target}")

        assert target.shape == transformed_frames[0].shape

        # Get prediction using 6 previous frames
        x = torch.cat(transformed_frames[:6], dim=1)
        pred = self.forward(x)

        return pred, target, transformed_frames[6], original_shape, transform_shape, transformed_frames, obj_mask

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

    def log_visualizations(self, clip_name, pred, target, gt, psnr, *frames):
        """Helper method to create and log visualizations"""
        fig, ax = plt.subplots(2, 6, figsize=(25, 10))
        
        # First row: Input sequence (6 frames)
        titles = ['Frame 1 (Current)', 'Frame 2 (Prev-1)', 'Frame 3 (Prev-2)', 
                 'Frame 4 (Prev-3)', 'Frame 5 (Prev-4)', 'Frame 6 (Prev-5)']
        for i in range(6):
            ax[0, i].imshow(frames[i][0].cpu().numpy(), cmap='gray')
            ax[0, i].set_title(titles[i])
        
        # Second row: Prediction, Target, Ground Truth, Metrics
        ax[1, 0].imshow(pred[0].cpu().numpy(), cmap='gray')
        ax[1, 0].set_title('Prediction')
        ax[1, 1].imshow(target[0].cpu().numpy(), cmap='gray')
        ax[1, 1].set_title('Target')
        ax[1, 2].imshow(gt[0].cpu().numpy(), cmap='gray')
        ax[1, 2].set_title('Ground Truth')
        
        metrics_text = f'GT SNR: {psnr.gt_snr_avg:.2f}\nPred SNR: {psnr.pred_snr_avg:.2f}\nRatio: {psnr.psnr_ratio:.2f}'
        ax[1, 3].text(0.1, 0.5, metrics_text, fontsize=14)
        ax[1, 3].axis('off')
        
        ax[1, 4].axis('off')
        ax[1, 5].axis('off')
        
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
                    f'psnr_plot_{clip_name}': wandb.Image(psnr.plot_comparison())
                })
            # otherwise, just rely on local save
        
        plt.close(fig)
        plt.close(psnr_fig)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
            },
        }