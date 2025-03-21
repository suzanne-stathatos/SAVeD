"""
Add the denoised images to the raw image.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

def add_images(denoised_path, raw_path, output_path):
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Get all subdirectories in raw path
    for clip_dir in os.listdir(raw_path):
        if not os.path.isdir(os.path.join(raw_path, clip_dir)):
            print(f"Skipping {clip_dir} because it is not a directory")
            continue
            
        # Create corresponding output subdirectory
        os.makedirs(os.path.join(output_path, clip_dir), exist_ok=True)
        
        # Process each frame
        raw_frames = os.path.join(raw_path, clip_dir)
        denoised_frames = os.path.join(denoised_path, clip_dir)
        
        for frame in tqdm(os.listdir(raw_frames), desc=f"Processing {clip_dir}"):
            if not frame.endswith(('.jpg', '.png', '.tif')):
                print(f"Skipping {frame} because it is not a jpg or png or tif")
                continue
                
            # Read images
            raw_img = cv2.imread(os.path.join(raw_frames, frame))
            denoised_img = cv2.imread(os.path.join(denoised_frames, frame))
            
            assert raw_img is not None, f"Raw image not found for {frame}"
            if denoised_img is None:
                # save raw image
                cv2.imwrite(os.path.join(output_path, clip_dir, frame), raw_img)
                continue    
            # else:
            # Take one channel from each image
            raw_img = raw_img[:, :, 0]
            denoised_img = denoised_img[:, :, 0]
            # Add extra channel dimension
            raw_img = np.expand_dims(raw_img, axis=2)
            denoised_img = np.expand_dims(denoised_img, axis=2)
                                
            # Add images together as 3 channel image with only 2 differences (raw, denoised, denoised)
            img_2_ch = np.concatenate([raw_img, denoised_img, denoised_img], axis=2)
            # print(img_2_ch.shape)
            
            # Save result
            output_file = os.path.join(output_path, clip_dir, frame)
            cv2.imwrite(output_file, img_2_ch)
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--denoised_imgs', required=True)
    parser.add_argument('--raw_imgs', required=True)
    parser.add_argument('--added_imgs', required=True)
    args = parser.parse_args()
    
    add_images(args.denoised_imgs, args.raw_imgs, args.added_imgs)

if __name__ == '__main__':
    main()