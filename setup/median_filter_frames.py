import numpy as np
import cv2
import os
from PIL import Image, ImageFile
from pathlib import Path
from scipy.ndimage import median_filter, gaussian_filter
from argparse import ArgumentParser


ImageFile.LOAD_TRUNCATED_IMAGES = True

def min_max_spatial_normalize(frames):
    '''
    Normalizes each frame to be between 0 and 1 temporally and spatially.
    '''
    frames = frames - np.min(frames)
    frames = frames / np.max(frames)

    frs = []
    for fr in frames:
        min_val = np.min(fr)
        max_val = np.max(fr)
        frs.append((fr - min_val) / (max_val - min_val))
    return np.stack(frs)


def get_median_filtered_frames(frames):
    '''
    Applies a median filter over time to the frames
    '''
    return median_filter(frames, size=3, mode='reflect', axes=0)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--frames-dir', type=str, required=True, help='Path to the raw frames')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to save the median filtered frames')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    domains = os.listdir(args.frames_dir)
    # domains = ["kenai-train"]

    for domain in domains:
        print(f'Processing domain {domain}')
        all_videos = [f for f in os.listdir(os.path.join(args.frames_dir, domain)) if os.path.isdir(os.path.join(args.frames_dir, domain, f)) and not f.startswith('.')]
        for video in all_videos:
            raw_video_path = os.path.join(args.frames_dir, domain, video)
            new_sub_video_path = os.path.join(args.output_dir, domain, video)
            # if os.path.exists(new_sub_video_path) and len(os.listdir(new_sub_video_path)) == len(os.listdir(raw_video_path)):
            #     print(f'Skipping {video} because it already exists')
            #     continue
            
            # Get all of the frames in the video
            raw_frame_names = sorted(os.listdir(raw_video_path), key=lambda x: int(x.split('.')[0]))
            raw_frames_paths = [os.path.join(raw_video_path, f) for f in raw_frame_names]
            # Open the frames as images
            raw_frame_imgs = np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in raw_frames_paths])

            # Get the median filtered frames
            median_filtered_frames = get_median_filtered_frames(raw_frame_imgs)

            # Min-max normalize the frames
            median_filtered_frames = min_max_spatial_normalize(median_filtered_frames)

            # Save the new image to the folder location 
            os.makedirs(new_sub_video_path, exist_ok=True)

            for i, frame_offset in enumerate(range(len(raw_frame_names))):
                combined_frame_img = np.dstack([
                                                median_filtered_frames[i], 
                                                median_filtered_frames[i], 
                                                median_filtered_frames[i]
                                                ]).astype(np.float32)
                # convert to uint8
                combined_frame_img = (combined_frame_img * 255).astype(np.uint8)
                Image.fromarray(combined_frame_img).save(os.path.join(new_sub_video_path, raw_frame_names[i]), quality=95)