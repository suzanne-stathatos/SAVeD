"""
Add the denoised images to the raw image.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

def add_images(channel1_path, channel2_path, channel3_path, output_path):
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Get all subdirectories in raw path
    for clip_dir in os.listdir(channel1_path):
        if not os.path.isdir(os.path.join(channel1_path, clip_dir)):
            continue
            
        # Create corresponding output subdirectory
        os.makedirs(os.path.join(output_path, clip_dir), exist_ok=True)
        
        # Process each frame
        channel1_frames = os.path.join(channel1_path, clip_dir)
        channel2_frames = os.path.join(channel2_path, clip_dir)
        channel3_frames = os.path.join(channel3_path, clip_dir)
        
        for frame in tqdm(os.listdir(channel1_frames), desc=f"Processing {clip_dir}"):
            if not frame.endswith(('.jpg', '.png')):
                continue
                
            # Read images
            channel1_img = cv2.imread(os.path.join(channel1_frames, frame))
            channel2_img = cv2.imread(os.path.join(channel2_frames, frame))
            channel3_img = cv2.imread(os.path.join(channel3_frames, frame))
            
            assert channel1_img is not None, f"Channel1 image not found for {frame}"
            if channel2_img is None:
                # save channel1 image
                cv2.imwrite(os.path.join(output_path, clip_dir, frame), channel1_img)
                continue    
            # else:
            # Take one channel from each image
            channel1_img = channel1_img[:, :, 0]
            channel2_img = channel2_img[:, :, 0]
            channel3_img = channel3_img[:, :, 0]
            # Add extra channel dimension
            channel1_img = np.expand_dims(channel1_img, axis=2)
            channel2_img = np.expand_dims(channel2_img, axis=2)
            channel3_img = np.expand_dims(channel3_img, axis=2)
                                
            # Add images together as 3 channel image with only 2 differences (raw, denoised, denoised)
            img_3_ch = np.concatenate([channel1_img, channel2_img, channel3_img], axis=2)
            # print(img_2_ch.shape)
            
            # Save result
            output_file = os.path.join(output_path, clip_dir, frame)
            cv2.imwrite(output_file, img_3_ch)
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel1', required=True)
    parser.add_argument('--channel2', required=True)
    parser.add_argument('--channel3', required=True)
    parser.add_argument('--added_imgs', required=True)
    args = parser.parse_args()
    
    add_images(args.channel1, args.channel2, args.channel3, args.added_imgs)

if __name__ == '__main__':
    main()