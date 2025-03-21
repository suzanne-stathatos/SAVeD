import os
import re
from pathlib import Path
from PIL import Image, ImageFile
import cv2

import torch
from torch.utils.data import Dataset
import numpy as np
from setup.make_background_subtracted_frame_to_frame_difference_frames import get_background_sub_frames

ImageFile.LOAD_TRUNCATED_IMAGES = True

class NoisyVideoDataset_prev_frames_5(Dataset):
    def __init__(self, base_folder, anno_folder, transform=None, target="l1"):
        self.base_folder = base_folder
        self.anno_folder = anno_folder
        self.transform = transform
        self.num_prev_frames = 5
        self.target = target
        self.image_files = self.load_image_files()
        if self.target == "bs_global":
            self.calculate_global_backgrounds()
        print(len(self.image_files))

    def load_image_files(self):
        # Walk through all subdirectories and collect image files
        image_files = []
        for root, _, files in os.walk(self.base_folder, followlinks=True):
            for file in files:
                if file.endswith('.jpg'):
                    image_files.append(os.path.join(root, file))

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
                                     os.path.join(base_path, f"{frame_number - i}.jpg") for i in range(1, self.num_prev_frames + 1)
                                 ] + [
                                     os.path.join(base_path, f"{frame_number + i}.jpg") for i in range(1, 4)
                                 ]

                # Add to filtered list only if all required files exist
                if all(os.path.exists(req_file) for req_file in required_files):
                    filtered_files.append(f)

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

    def get_clip_anno_path(self, img_path):
        clip = Path(os.path.dirname(img_path)).name
        anno_path = os.path.join(self.anno_folder, clip, 'gt.txt')
        assert os.path.exists(anno_path), f"Annotation file not found at {anno_path}"
        return anno_path

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
        prev_3_img_path = self.get_previous_frame_path(prev_2_img_path)
        prev_4_img_path = self.get_previous_frame_path(prev_3_img_path)
        prev_5_img_path = self.get_previous_frame_path(prev_4_img_path)
        future_img_path = self.get_future_frame_path(img_path)
        prev_future_img_path = self.get_previous_frame_path(future_img_path)

        frame1 = Image.open(img_path).convert('L')
        frame2 = Image.open(prev_img_path).convert('L')
        frame3 = Image.open(prev_2_img_path).convert('L')
        frame4 = Image.open(prev_3_img_path).convert('L')
        frame5 = Image.open(prev_4_img_path).convert('L')
        frame6 = Image.open(prev_5_img_path).convert('L')
        frame7 = Image.open(future_img_path).convert('L')
        frame8 = Image.open(prev_future_img_path).convert('L')
        # Create empty Image same size as frame1
        obj_mask = Image.new('L', frame1.size)

        if self.anno_folder:
            anno_path = self.get_clip_anno_path(img_path)
            with open(anno_path, 'r') as f:
                all_annos = f.readlines()
            # get annotations for current img_path
            frame_id = int(re.findall(r'(\d+)\.jpg', img_path)[0])
            annos = [anno for anno in all_annos if int(anno.split(',')[0]) == frame_id+1] # +1 because MOT is 1-indexed
            # mot annos are frame_id, track_id, x, y, w, h, conf, x1, y1, z1
            for anno in annos:
                _, _, x, y, w, h, conf, _, _, _ = map(int, map(float, anno.split(',')))
                obj_mask.paste(1, (x, y, x+w, y+h))
        else:
            # make the obj mask all 1s so no penalty is applied
            obj_mask.paste(1, (0,0, obj_mask.width, obj_mask.height))

        if self.transform:
            frame1 = self.transform(frame1)
            frame2 = self.transform(frame2)
            frame3 = self.transform(frame3)
            frame4 = self.transform(frame4)
            frame5 = self.transform(frame5)
            frame6 = self.transform(frame6)
            frame7 = self.transform(frame7)
            frame8 = self.transform(frame8)
            obj_mask = self.transform(obj_mask)

        device = frame1.device
        if self.target == "l1":
            target = torch.abs(frame7 - frame8)
        elif self.target == "l1_and_obj":
            target = torch.abs(frame7 - frame8)
        elif self.target == "l2":
            target = (frame7 - frame8) ** 2
        elif self.target == "bs_local":
            bs_frames = get_background_sub_frames(np.stack([frame6, frame5, frame4, frame3, frame2, frame1, frame7, frame8]))
            target = torch.tensor(bs_frames[5])
        elif self.target == "bs_global":
            clip = os.path.dirname(img_path)
            target = frame1 - self.global_backgrounds[clip]
        elif self.target == "stddev_all":
            frame_next2 = Image.open(self.get_future_frame_path(future_img_path)).convert('L')
            if self.transform:
                frame_next2 = self.transform(frame_next2)
            target = torch.std(torch.stack([frame6, frame5, frame4, frame3, frame2, frame1, frame7, frame_next2]), dim=0)
        elif self.target == "stddev_3":
            target = torch.std(torch.stack([frame2, frame1, frame7]), dim=0)
        else:
            raise ValueError(f"Invalid target: {self.target}")
        target = target.to(device)
        return frame1, frame2, frame3, frame4, frame5, frame6, target, obj_mask

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

