from collections import defaultdict
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/suzanne/SAVeD')
from metrics.crops import get_crops

from skimage import io, img_as_float
from skimage.metrics import mean_squared_error
from skimage.restoration import estimate_sigma


class PSNR_classical:
    def __init__(self, pred_img_path, bboxes, clip_name, frame_idx):
        self.pred_img_path = pred_img_path
        self.bboxes = bboxes
        self.clip_name = clip_name
        self.frame_idx = frame_idx
    
    def calculate_psnr_image_vs_bw(self):
        # Calculate psuedo psnr for the entire image
        pred_img = img_as_float(io.imread(self.pred_img_path))
        # print(pred_img.shape)

        # create an all black image of the same size
        black_img = np.zeros_like(pred_img)
        # add white rectangles to the image where the bboxes are
        for bbox in self.bboxes:
            x1, y1, w, h = bbox
            black_img[y1:y1+h, x1:x1+w] = 1
        # calculate the psnr between the black_img and the pred_img
        psnr = cv2.PSNR(black_img, pred_img, 1.0)
        return psnr
    
    def calculate_psnr_image_vs_target(self, target_img):
        # calculate the psnr between the target_img and the pred_img
        # convert target_img to numpy array
        target_img = target_img.permute(1, 2, 0)
        # remove the last dimension
        target_img = target_img[:, :, 0]
        target_img = np.array(target_img)
        if np.max(target_img) > 1:
            # normalize the target image to be between 0 and 1
            target_img = (target_img - np.min(target_img)) / (np.max(target_img) - np.min(target_img))
        # load pred_img_path as a numpy array
        pred_img = img_as_float(io.imread(self.pred_img_path))
        # convert pred_img to numpy array
        pred_img = np.array(pred_img)
        if len(pred_img.shape) > 2:
            # remove the last dimension
            pred_img = pred_img[:, :, 0]
        pred_img = pred_img.astype(np.float32)
        target_img = target_img.astype(np.float32)
        assert target_img.shape == pred_img.shape, f"Shape mismatch: {target_img.shape} vs {pred_img.shape}"
        psnr = cv2.PSNR(target_img, pred_img, 1.0)
        return psnr

    def calculate_pseudo_psnr_boxes(self):
        # Calculate psnr for each bbox and return the average
        pass



class PSNR:
    def __init__(self, gt_img, pred_img, bboxes, clip_name, frame_idx):
        self.gt_img = gt_img
        self.pred_img = pred_img
        self.bboxes = bboxes
        self.num_locations = 10
        self.clip_name = clip_name
        self.frame_idx = frame_idx


    def get_snr(self, signal_crops, all_noise_crops):
        """Vectorized SNR calculation"""
        snr_values = []
        
        for signal_crop, noise_crops in zip(signal_crops, all_noise_crops):
            # Calculate signal power once
            signal_power = np.maximum(np.mean(signal_crop ** 2), 1e-10)
            
            # Calculate all noise powers at once
            noise_powers = np.maximum(np.array([np.mean(n ** 2) for n in noise_crops]), 1e-10)
            
            # Calculate all SNRs at once
            snrs = 10 * np.log10(signal_power / noise_powers)
            
            # Filter out any invalid values
            valid_snrs = snrs[~np.isnan(snrs)]
            if len(valid_snrs) > 0:
                snr_values.append(np.mean(valid_snrs))
        
        return np.mean(snr_values) if snr_values else 0

    def ratio(self):
        if len(self.bboxes) == 0:
            self.psnr_ratio = -1
            self.gt_snr_avg = -1
            self.pred_snr_avg = -1
            self.random_locations_list = []
            self.gt_noise_crops_list = []
            self.pred_noise_crops_list = []
            return -1
        
        # Preallocate lists
        gt_crops, pred_crops, gt_noise_crops_list, pred_noise_crops_list, random_locations_list = get_crops ( self.gt_img, 
                                                                                                             self.pred_img, 
                                                                                                             self.bboxes, 
                                                                                                             num_noise_crops_per_bbox=10)

        if len(gt_crops) == 0:
            self.psnr_ratio = 0
            self.gt_snr_avg = 0
            self.pred_snr_avg = 0
            self.random_locations_list = []
            self.gt_noise_crops_list = []
            self.pred_noise_crops_list = []
            return 0
        
        # Calculate SNR values
        gt_snr_avg = self.get_snr(gt_crops, gt_noise_crops_list)
        pred_snr_avg = self.get_snr(pred_crops, pred_noise_crops_list)
        
        # Store results
        self.pred_snr_avg = pred_snr_avg
        self.gt_snr_avg = gt_snr_avg
        self.psnr_ratio = pred_snr_avg - gt_snr_avg
        self.random_locations_list = random_locations_list
        self.gt_noise_crops_list = gt_noise_crops_list
        self.pred_noise_crops_list = pred_noise_crops_list
        
        return self.psnr_ratio

    def plot_comparison(self):
        """Create a comparison plot showing GT, prediction, and random crop locations"""
        fig, axes = plt.subplots(1, 2, figsize=(10, 10))
        
        # Convert images to RGB for drawing
        gt_rgb = cv2.cvtColor((self.gt_img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        pred_rgb = cv2.cvtColor((self.pred_img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        # Draw bounding boxes and random locations for each bbox
        for bbox_idx, (x1, y1, w, h) in enumerate(self.bboxes):
            # Draw main bbox
            cv2.rectangle(gt_rgb, (int(x1), int(y1)), (int(x1+w), int(y1+h)), (255, 255, 255), 2)
            cv2.rectangle(pred_rgb, (int(x1), int(y1)), (int(x1+w), int(y1+h)), (255, 255, 255), 2)
            
            # Draw random crop locations for this bbox
            random_locations = self.random_locations_list[bbox_idx]
            noise_crops = self.gt_noise_crops_list[bbox_idx]
            for (y, x), noise_crop in zip(random_locations, noise_crops):
                crop_h, crop_w = noise_crop.shape
                cv2.rectangle(gt_rgb, (int(x), int(y)), 
                            (int(x+crop_w), int(y+crop_h)), (0, 0, 255), 1)
                cv2.rectangle(pred_rgb, (int(x), int(y)), 
                            (int(x+crop_w), int(y+crop_h)), (0, 0, 255), 1)
        
        # Add metrics text
        gt_text = f"GT SNR: {self.gt_snr_avg:.2f}"
        pred_text = f"Pred SNR: {self.pred_snr_avg:.2f}\nRatio: {self.psnr_ratio:.2f}"
        
        cv2.putText(gt_rgb, gt_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(pred_rgb, pred_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display images
        axes[0].imshow(gt_rgb)
        axes[0].set_title('Ground Truth')
        axes[1].imshow(pred_rgb)
        axes[1].set_title('Prediction')
        
        plt.tight_layout()
        return fig