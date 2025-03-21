import os
from pathlib import Path
import re
from PIL import Image, ImageFile, ImageDraw, ImageFont
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from einops import rearrange

from torch.utils.data import Dataset

from setup.make_background_subtracted_frame_to_frame_difference_frames import get_background_sub_frames

ImageFile.LOAD_TRUNCATED_IMAGES = True

def verify_paste(original_img, pasted_img, save_path):
    """
    Create a side-by-side comparison of original and pasted images
    
    Args:
        original_img: PIL Image or numpy array
        pasted_img: PIL Image or numpy array
        save_path: Path to save the comparison image
    """
    # Convert to PIL if numpy
    if isinstance(original_img, np.ndarray):
        original_img = Image.fromarray(original_img)
    if isinstance(pasted_img, np.ndarray):
        pasted_img = Image.fromarray(pasted_img)
    
    # Create side-by-side image
    total_width = original_img.width * 2
    max_height = max(original_img.height, pasted_img.height)
    comparison = Image.new('RGB', (total_width, max_height))
    
    # Paste images
    comparison.paste(original_img, (0, 0))
    comparison.paste(pasted_img, (original_img.width, 0))
    
    # Add dividing line
    draw = ImageDraw.Draw(comparison)
    draw.line([(original_img.width, 0), (original_img.width, max_height)], 
              fill='red', width=2)
    
    # Add labels
    font = ImageFont.load_default()
    draw.text((10, 10), "Original", fill='white', font=font)
    draw.text((original_img.width + 10, 10), "Pasted", fill='white', font=font)
    
    # Save
    comparison.save(save_path)
    return comparison


class NoisyVideoDatasetWithWindowSize(Dataset):
    def __init__(self, base_folder, transform=None, target="l1", anno_folder=None, window_size=31):
        self.base_folder = base_folder
        self.transform = transform
        self.target = target
        self.anno_folder = anno_folder
        self.window_size = window_size
        self.image_files = self.load_image_files()
        if target == "bs_global" or target == "sum_minus_3_background" or target == "sum_minus_5_background":
            self.calculate_global_backgrounds()
        self.future_offset = 5
        print(len(self.image_files))

    def load_image_files(self):
        # print(f"Checking base folder: {self.base_folder}")
        # print(f"Base folder exists: {os.path.exists(self.base_folder)}")
        # print(f"Base folder contents: {os.listdir(self.base_folder)}")

        # Walk through all subdirectories and collect image files
        image_files = []
        for root, dirs, files in os.walk(self.base_folder, followlinks=True):
            # print(f"Checking directory: {root}")
            # print(f"Contains directories: {dirs}")
            # print(f"Contains files: {len(files)}")
        
            for file in files:
                if file.endswith('.jpg'):
                    image_files.append(os.path.join(root, file))
        
        # print(f"Initial number of jpg files found: {len(image_files)}")
        if len(image_files) == 0:
            print("No jpg files found! Check path and file extensions.")
            return []

        # Filter only for files that have adjacent frames (previous and next)
        filtered_files = []
        for f in image_files:
            # Extract the numeric part of the filename
            match = re.search(r'(\d+)\.jpg$', f)
            if match:
                frame_number = int(match.group(1))
                base_path = os.path.dirname(f)

                # Check if previous and next frames exist
                required_files = [
                    os.path.join(base_path, f"{frame_number - i}.jpg") for i in range(1, self.window_size//2+1)
                ] + [
                    os.path.join(base_path, f"{frame_number + i}.jpg") for i in range(1, self.window_size//2+1)
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
        all_img_paths = []
        img_path = self.image_files[idx]
        prev_img_path = self.get_previous_frame_path(img_path)
        all_img_paths.append(prev_img_path)
        for i in range(0, self.window_size//2-1):
            prev_img_path = self.get_previous_frame_path(prev_img_path)
            all_img_paths.append(prev_img_path)
        # reverse the list
        all_img_paths = all_img_paths[::-1]
        # add the current frame
        all_img_paths.append(img_path)
        # add the future frames
        future_img_path = self.get_future_frame_path(img_path)
        all_img_paths.append(future_img_path)
        for i in range(self.window_size//2-1):
            future_img_path = self.get_future_frame_path(future_img_path)
            all_img_paths.append(future_img_path)
        # all_img_paths should now be in the correct order prev to curr to future
        # print(all_img_paths)
        # open all paths as PIL images
        frames = [Image.open(img_path).convert('L') for img_path in all_img_paths]
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        device = frames[0].device

        curr_idx = self.window_size//2
        if self.target == "l1":
            target = torch.abs(frames[curr_idx] - frames[curr_idx+1])
        elif self.target == "sum_minus_3_background":
            clip = os.path.dirname(img_path)
            background = Image.fromarray(self.global_backgrounds[clip].astype(np.uint8))
            background = self.transform(background)
            target_non_abs = frames[curr_idx] + frames[curr_idx-1] + frames[curr_idx+1] - 3*background
            target_positive = torch.where(target_non_abs > 0, target_non_abs, torch.zeros_like(target_non_abs))
            target = target_positive
        elif self.target == "sum_minus_5_background":
            clip = os.path.dirname(img_path)
            background = Image.fromarray(self.global_backgrounds[clip].astype(np.uint8))
            background = self.transform(background)
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
            bs_frames = get_background_sub_frames(np.stack(frames))
            target = torch.tensor(bs_frames[curr_idx])
        elif self.target == "bs_global":
            clip = os.path.dirname(img_path)
            background = Image.fromarray(self.global_backgrounds[clip].astype(np.uint8))
            background = self.transform(background)
            target = frames[curr_idx] - background
        elif self.target == "stddev_all":
            target = torch.std(frames, dim=0)
            assert target.shape == frames[curr_idx].shape
        elif self.target == "stddev_3":
            target = torch.std(torch.stack([frames[curr_idx-1], frames[curr_idx], frames[curr_idx+1]]), dim=0)
            assert target.shape == frames[curr_idx].shape
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

        # normalize between 0 and 1
        frames = torch.stack(frames)
        frames -= torch.min(frames)
        frames /= torch.max(frames)

        target -= torch.min(target)
        target /= torch.max(target)

        frames = rearrange(frames, 't c h w -> c h w t')
        # target = rearrange(target, 't h w c -> c h w t')
        return frames, curr_idx, target

    def get_clip_anno_path(self, img_path):
        clip = Path(os.path.dirname(img_path)).name
        anno_path = os.path.join(self.anno_folder, clip, 'gt.txt')
        assert os.path.exists(anno_path), f"Annotation file not found at {anno_path}"
        return anno_path

    def get_future_frame_path(self, img_path, offset=1):
        frame_num = int(re.findall(r'(\d+)\.jpg', img_path)[0])
        future_frame_num = frame_num + offset
        future_img_path = img_path.replace(f'{frame_num}.jpg', f'{future_frame_num}.jpg')
        return future_img_path

    def get_previous_frame_path(self, img_path):
        frame_num = int(re.findall(r'(\d+)\.jpg', img_path)[0])
        prev_frame_num = frame_num - 1
        prev_img_path = img_path.replace(f'{frame_num}.jpg', f'{prev_frame_num}.jpg')
        return prev_img_path

    def detect_motion(self, frame1, frame2):
        diff = torch.abs(frame1 - frame2)
        motion_mask = diff > 0.05  # Adjust threshold as needed
        return motion_mask.float()

    def random_mask(self, img):
        # mask = torch.rand_like(img) > 0.95
        mask = torch.rand_like(img) > 0.95
        masked_img = img.clone()
        masked_img[mask] = 0
        return masked_img, mask
