import numpy as np
import cv2
import os
from PIL import Image, ImageFile
from pathlib import Path
from setup.background_subtract_frames import get_background_sub_frames
from setup.l1_frames import get_frame_l1_diff_frames

ImageFile.LOAD_TRUNCATED_IMAGES = True


def generate_and_save_background_sub_frame_to_frame_diff_frames(raw_frames_path, background_sub_frames_path):
    # Get all of the frames in the video
    # print(f'Processing video {Path(raw_frames_path).name}')
    raw_frame_names = sorted(os.listdir(raw_frames_path), key=lambda x: int(x.split('.')[0]))
    # print(raw_frame_names)
    raw_frames_paths = [os.path.join(raw_frames_path, f) for f in raw_frame_names]

    # Open the frames as images
    raw_frame_imgs = np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in raw_frames_paths])

    # Get the background subtracted frames
    background_sub_frame_imgs = get_background_sub_frames(raw_frame_imgs)

    # Get the frame to frame difference frames
    frame_to_frame_diff_imgs = get_frame_l1_diff_frames(background_sub_frame_imgs)

    # Save the combined image to the folder location 
    os.makedirs(background_sub_frames_path, exist_ok=True)

    for i, frame_offset in enumerate(range(len(raw_frame_names))):
        combined_frame_img = np.dstack([
                                        raw_frame_imgs[i] / 255, 
                                        background_sub_frame_imgs[i], 
                                        frame_to_frame_diff_imgs[i]
                                        ]).astype(np.float32)
        # convert to uint8
        combined_frame_img = (combined_frame_img * 255).astype(np.uint8)
        Image.fromarray(combined_frame_img).save(os.path.join(background_sub_frames_path, raw_frame_names[i]), quality=95)


if __name__ == "__main__":
    CFC22_FRAMES_PATH = '/path/to/Data/CFC22/frames/raw'
    CFC22_FRAMES_BACKGROUND_SUB_PATH = '/path/to/Data/CFC22/frames/raw_bs'

    domains = os.listdir(CFC22_FRAMES_PATH)

    for domain in domains:
        print(f'Processing domain {domain}')
        all_videos = [f for f in os.listdir(os.path.join(CFC22_FRAMES_PATH, domain)) if os.path.isdir(os.path.join(CFC22_FRAMES_PATH, domain, f)) and not f.startswith('.')]
        for video in all_videos:
            raw_video_path = os.path.join(CFC22_FRAMES_PATH, domain, video)
            background_sub_video_path = os.path.join(CFC22_FRAMES_BACKGROUND_SUB_PATH, domain, video)
            if os.path.exists(background_sub_video_path) and len(os.listdir(background_sub_video_path)) == len(os.listdir(raw_video_path)):
                print(f'Skipping {video} because it already exists')
                continue

            generate_and_save_background_sub_frame_to_frame_diff_frames(raw_video_path, background_sub_video_path)

    # DEBUGGING
    # raw_video_path = '/path/to/denoised_v1/samples/kenai-channel/2018-08-16-JD228_Channel_Stratum1_Set1_CH_2018-08-16_060006_532_732'
    # background_sub_video_path = '/path/to/denoised_v1/samples/kenai-channel/bg_sub_2018-08-16-JD228_Channel_Stratum1_Set1_CH_2018-08-16_060006_532_732'
    # generate_and_save_background_sub_frame_to_frame_diff_frames(raw_video_path, background_sub_video_path)
