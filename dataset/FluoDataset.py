import os
from pathlib import Path
import re
from PIL import Image, ImageFile, ImageDraw, ImageFont
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import re
from setup.make_background_subtracted_frame_to_frame_difference_frames import get_background_sub_frames

from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

def format_frame_path(base_path, frame_num):
    return os.path.join(base_path, f"t{frame_num:03d}.tif")


class FluoDataset(Dataset):
    def __init__(self, base_folder, transform=None, target="l1", window_size=5):
        self.base_folder = base_folder
        self.transform = transform
        self.target = target
        self.window_size = window_size
        self.image_files = self.load_image_files()
        if target == "bs_global" or target == "sum_minus_3_background" or target == "sum_minus_5_background":
            self.calculate_global_backgrounds()
        self.future_offset = 5
        print(len(self.image_files))

    def load_image_files(self):
        # Walk through all subdirectories and collect image files
        image_files = []
        for root, dirs, files in os.walk(self.base_folder, followlinks=True):
            for file in files:
                if file.endswith('.tif'):
                    image_files.append(os.path.join(root, file))
        
        # print(f"Initial number of png files found: {len(image_files)}")
        if len(image_files) == 0:
            print("No jpg files found! Check path and file extensions.")
            return []

        # Filter only for files that have adjacent frames (previous and next)
        filtered_files = []
        for f in image_files:
            # Extract the numeric part of the filename
            match = re.search(r'(\d+)\.tif$', f)
            if match:
                frame_number = int(match.group(1))
                base_path = os.path.dirname(f)

                # Check if previous and next frames exist
                required_files = [
                    format_frame_path(base_path, frame_number - i) for i in range(1, 4)
                ] + [
                    format_frame_path(base_path, frame_number + i) for i in range(1, 4)
                ]

                # Add to filtered list only if all required files exist
                if all(os.path.exists(req_file) for req_file in required_files):
                    filtered_files.append(f)
                else:
                    missing = [req for req in required_files if not os.path.exists(req)]
                    # print(f"Skipping {f} due to missing required files: {missing}")

        print(f"Number of files after filtering: {len(filtered_files)}")
        if len(filtered_files) == 0:
            print("WARNING: No files passed the filtering criteria!")
            # Print a few example paths to verify correct directory structure
            print(f"Example paths checked: {image_files[:5]}")
        
        return filtered_files

    
    def calculate_global_backgrounds(self):
        # Initialize dict of clip to global background
        self.global_backgrounds = {}
        # Get all unique clips from image files
        clips = [os.path.dirname(f) for f in self.image_files]
        unique_clips = list(set(clips))
        # Calculate global background for each clip
        for clip in unique_clips:
            clip_frames = [f for f in self.image_files if os.path.dirname(f) == clip]
            frames = np.stack([Image.open(f).convert('L') for f in clip_frames])
            blurred_frames = frames.astype(np.float32)
            # Blur frames
            for i in range(frames.shape[0]):
                blurred_frames[i] = cv2.GaussianBlur(blurred_frames[i], (5,5), 0)
            self.global_backgrounds[clip] = np.mean(blurred_frames, axis=0)

    def has_previous_frame(self, img_path):
        prev_img_path = self.get_previous_frame_path(img_path)
        return os.path.exists(prev_img_path)

    def has_future_frame(self, img_path):
        future_img_path = self.get_future_frame_path(img_path)
        return os.path.exists(future_img_path)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        prev_img_path = self.get_previous_frame_path(img_path)
        prev_2_img_path = self.get_previous_frame_path(prev_img_path)
        future_img_path = self.get_future_frame_path(img_path)
        prev_future_img_path = self.get_previous_frame_path(future_img_path)

        frame1 = Image.open(img_path).convert('L')
        frame2 = Image.open(prev_img_path).convert('L')
        frame3 = Image.open(prev_2_img_path).convert('L')
        frame4 = Image.open(future_img_path).convert('L')
        frame5 = Image.open(prev_future_img_path).convert('L')
        # Create empty Image same size as frame1
        obj_mask = Image.new('L', frame1.size)

        if self.transform:
            frame1 = self.transform(frame1)
            frame2 = self.transform(frame2)
            frame3 = self.transform(frame3)
            frame4 = self.transform(frame4)
            frame5 = self.transform(frame5)
            obj_mask = self.transform(obj_mask)
        
        device = frame1.device
        if self.target == "l1":
            target = torch.abs(frame4 - frame5)
        elif self.target == "sum_minus_3_background":
            clip = os.path.dirname(img_path)
            background = Image.fromarray(self.global_backgrounds[clip].astype(np.uint8))
            background = self.transform(background)
            target_non_abs = frame1 + frame2 + frame3 - 3*background
            target_positive = torch.where(target_non_abs > 0, target_non_abs, torch.zeros_like(target_non_abs))
            target = target_positive
        elif self.target == "sum_minus_5_background":
            clip = os.path.dirname(img_path)
            background = Image.fromarray(self.global_backgrounds[clip].astype(np.uint8))
            background = self.transform(background)
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
            future_diff = frame4 - frame1 
            future_diff = torch.where(future_diff > 0, future_diff, torch.zeros_like(future_diff))
            prev_diff = frame2 - frame1 # exaggerates the motion in the backward direction
            prev_diff = torch.where(prev_diff > 0, prev_diff, torch.zeros_like(prev_diff))
            target = frame1 + future_diff + prev_diff
        elif self.target == "gap1_inverted_and_curr_frame":
            future_diff = frame4 - frame1 
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
        elif self.target == "l2":
            target = (frame4 - frame5) ** 2
        elif self.target == "bs_local":
            bs_frames = get_background_sub_frames(np.stack([frame3, frame2, frame1, frame4, frame5]))
            target = torch.tensor(bs_frames[2])
        elif self.target == "bs_global":
            clip = os.path.dirname(img_path)
            background = Image.fromarray(self.global_backgrounds[clip].astype(np.uint8))
            background = self.transform(background)
            target = frame1 - background
        elif self.target == "stddev_all":
            frame_next2 = Image.open(self.get_future_frame_path(future_img_path)).convert('L')
            if self.transform:
                frame_next2 = self.transform(frame_next2)
            target = torch.std(torch.stack([frame3, frame2, frame1, frame4, frame_next2]), dim=0)
            assert target.shape == frame1.shape
        elif self.target == "stddev_3":
            target = torch.std(torch.stack([frame2, frame1, frame4]), dim=0)
            assert target.shape == frame1.shape
        else: 
            raise ValueError(f"Target {self.target} not supported")
        
        # matplotlib plot it
        target = target.to(device)
        # fig, ax = plt.subplots(1, 4)
        # ax[0].imshow(frame1[0].cpu().numpy(), cmap='gray')
        # ax[1].imshow(frame2[0].cpu().numpy(), cmap='gray')
        # ax[2].imshow(frame3[0].cpu().numpy(), cmap='gray')
        # ax[3].imshow(target[0].cpu().numpy(), cmap='gray')
        # # plt.show()
        # plt.savefig(f"{self.target}.png")
        return frame1, frame2, frame3, target, obj_mask

    def get_future_frame_path(self, img_path, offset=1):
        frame_num = int(re.findall(r't(\d+)\.tif', img_path)[0])
        future_frame_num = frame_num + offset
        future_img_path = img_path.replace(f't{frame_num:03d}.tif', f't{future_frame_num:03d}.tif')
        return future_img_path

    def get_previous_frame_path(self, img_path):
        frame_num = int(re.findall(r't(\d+)\.tif', img_path)[0])
        prev_frame_num = frame_num - 1
        prev_img_path = img_path.replace(f't{frame_num:03d}.tif', f't{prev_frame_num:03d}.tif')
        return prev_img_path

    def detect_motion(self, frame1, frame2):
        diff = torch.abs(frame1 - frame2)
        motion_mask = diff > 0.05  # Adjust threshold as needed
        return motion_mask.float()

