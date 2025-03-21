import numpy as np
import cv2
import os
from PIL import Image, ImageFile
from argparse import ArgumentParser
from setup.background_subtract_frames import get_background_sub_frames

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_frame_l1_diff_frames(frames):
    '''
    Computes the L1 difference between each frame and the next frame.
    Normalizes the result to [0,1]
    '''
    # Calculate differences
    l1_diff_frames = np.abs(frames[1:] - frames[:-1])
    
    # Duplicate last difference for final frame
    last_diff = np.abs(frames[-1] - frames[-2])
    l1_diff_frames = np.concatenate([l1_diff_frames, last_diff[np.newaxis, ...]], axis=0)
    
    return l1_diff_frames


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--frames-dir', type=str, required=True, help='Path to the raw frames')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to save the l1 frames')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    domains = os.listdir(args.frames_dir)

    for domain in domains:
        print(f'Processing domain {domain}')
        all_videos = [f for f in os.listdir(os.path.join(args.frames_dir, domain)) if os.path.isdir(os.path.join(args.frames_dir, domain, f)) and not f.startswith('.')]
        for video in all_videos:
            raw_video_path = os.path.join(args.frames_dir, domain, video)
            new_video_path = os.path.join(args.output_dir, domain, video)
            if os.path.exists(new_video_path) and len(os.listdir(new_video_path)) == len(os.listdir(raw_video_path)):
                print(f'Skipping {video} because it already exists')
                continue
            
            # Get all of the frames in the video
            raw_frame_names = sorted(os.listdir(raw_video_path), key=lambda x: int(x.split('.')[0]))
            raw_frames_paths = [os.path.join(raw_video_path, f) for f in raw_frame_names]
            # Open the frames as images
            raw_frame_imgs = np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in raw_frames_paths])
            background_sub_frame_imgs = get_background_sub_frames(raw_frame_imgs)
            
            # Get the l1 frames
            l1_frames = get_frame_l1_diff_frames(background_sub_frame_imgs)
            # Save the combined image to the folder location 
            os.makedirs(new_video_path, exist_ok=True)

            for i, frame_offset in enumerate(range(len(raw_frame_names))):
                combined_frame_img = np.dstack([
                                                l1_frames[i], 
                                                l1_frames[i], 
                                                l1_frames[i]
                                                ]).astype(np.float32)
                # convert to uint8
                combined_frame_img = (combined_frame_img * 255).astype(np.uint8)
                Image.fromarray(combined_frame_img).save(os.path.join(new_video_path, raw_frame_names[i]), quality=95)
