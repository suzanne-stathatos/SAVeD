import numpy as np
import cv2
import os
from PIL import Image, ImageFile
from argparse import ArgumentParser
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_background_sub_frames(frames):
    '''
    Blurs each frame, then subtracts the mean of the blurred frames from each frame.
    Normalizes the result to be between 0 and 1
    '''
    # Convert to float32 for better precision
    blurred_frames = frames.astype(np.float32)
    
    # Blur frames
    for i in range(frames.shape[0]):
        blurred_frames[i] = cv2.GaussianBlur(blurred_frames[i], (5,5), 0)
    
    # Subtract mean
    mean_frame = np.mean(blurred_frames, axis=0)
    background_sub_frames = frames - mean_frame  # Subtract from original frames, not blurred
    
    mean_normalization_value = np.max(np.abs(background_sub_frames))
    background_sub_frames /= mean_normalization_value
    background_sub_frames += 1
    background_sub_frames /= 2    
    return background_sub_frames


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--frames-dir', type=str, required=True, help='Path to the raw frames')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to save the background subtracted frames')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    domains = os.listdir(args.frames_dir)

    for domain in domains:
        print(f'Processing domain {domain}')
        all_videos = [f for f in os.listdir(os.path.join(args.frames_dir, domain)) if os.path.isdir(os.path.join(args.frames_dir, domain, f)) and not f.startswith('.')]
        for video in all_videos:
            raw_video_path = os.path.join(args.frames_dir, domain, video)
            background_sub_video_path = os.path.join(args.output_dir, domain, video)
            if os.path.exists(background_sub_video_path) and len(os.listdir(background_sub_video_path)) == len(os.listdir(raw_video_path)):
                print(f'Skipping {video} because it already exists')
                continue
            
            # Get all of the frames in the video
            try:
                raw_frame_names = sorted(os.listdir(raw_video_path), key=lambda x: int(x.split('.')[0]))
            except Exception as e:
                print(f'Error getting frames for {raw_video_path}: {e}')
                exit()
            raw_frames_paths = [os.path.join(raw_video_path, f) for f in raw_frame_names]
            # Open the frames as images
            raw_frame_imgs = np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in raw_frames_paths])
            # Get the background subtracted frames
            background_sub_frame_imgs = get_background_sub_frames(raw_frame_imgs)
            # Save the combined image to the folder location 
            os.makedirs(background_sub_video_path, exist_ok=True)

            for i, frame_offset in enumerate(range(len(raw_frame_names))):
                combined_frame_img = np.dstack([
                                                background_sub_frame_imgs[i], 
                                                background_sub_frame_imgs[i], 
                                                background_sub_frame_imgs[i]
                                                ]).astype(np.float32)
                # convert to uint8
                combined_frame_img = (combined_frame_img * 255).astype(np.uint8)
                Image.fromarray(combined_frame_img).save(os.path.join(background_sub_video_path, raw_frame_names[i]), quality=95)