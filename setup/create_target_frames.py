import numpy as np
import cv2
import os
from PIL import Image, ImageFile
from argparse import ArgumentParser
from tqdm import tqdm
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True

def save_gap1_frames(frames, args, video_name):
    '''
    Computes the positive difference in the backward direction and the positive difference in the forward direction.
    '''
    future_diff = frames[1:] - frames[:-1]
    future_diff = np.where(future_diff > 0, future_diff, np.zeros_like(future_diff))
    # make future_diff same shape as frames by adding a dimension to the back
    future_diff = np.concatenate([future_diff, np.zeros((1, frames.shape[1], frames.shape[2]))], axis=0)

    prev_diff = frames[:-1] - frames[1:] 
    # make prev_diff same shape as frames by adding a dimension to the front
    prev_diff = np.where(prev_diff > 0, prev_diff, np.zeros_like(prev_diff))
    prev_diff = np.concatenate([np.zeros((1, frames.shape[1], frames.shape[2])), prev_diff], axis=0)

    assert future_diff.shape == frames.shape, f"future_diff shape: {future_diff.shape}, frames shape: {frames.shape}"
    assert prev_diff.shape == frames.shape, f"prev_diff shape: {prev_diff.shape}, frames shape: {frames.shape}"
    gap1_frames = frames + future_diff + prev_diff

    # save each of the frames separately
    output_dir = os.path.join(args.output_dir, args.domain, video_name)
    # output_dir = 'test'
    os.makedirs(output_dir, exist_ok=True)
    for i in range(len(gap1_frames)):
        Image.fromarray(gap1_frames[i].astype(np.uint8), mode='L').save(os.path.join(output_dir, f'{i}.jpg'), quality=95)


def save_gap2_frames(frames, args, video_name):
    future_diff = frames[2:] - frames[:-2]
    future_diff = np.where(future_diff > 0, future_diff, np.zeros_like(future_diff))
    future_diff = np.concatenate([future_diff, np.zeros((2, frames.shape[1], frames.shape[2]))], axis=0)

    prev_diff = frames[:-2] - frames[2:]
    prev_diff = np.where(prev_diff > 0, prev_diff, np.zeros_like(prev_diff))
    prev_diff = np.concatenate([np.zeros((2, frames.shape[1], frames.shape[2])), prev_diff], axis=0)

    assert future_diff.shape == frames.shape
    assert prev_diff.shape == frames.shape
    gap2_frames = frames + future_diff + prev_diff

    output_dir = os.path.join(args.output_dir, args.domain, video_name)
    # output_dir = 'test'
    os.makedirs(output_dir, exist_ok=True)
    for i in range(len(gap2_frames)):
        Image.fromarray(gap2_frames[i].astype(np.uint8), mode='L').save(os.path.join(output_dir, f'{i}.jpg'), quality=95)


def save_stddev_5_frames(frames, args, video_name):
    # get 5 frames at a time centered on the current frame, with a stride of 1
    stddev_frames = []
    for i in range(2, len(frames) - 2):
        stddev_frames.append(np.std(frames[i-2:i+3], axis=0))
    # add the raw frames to the front of the list and the back of the list
    stddev_frames = np.array(stddev_frames)
    stddev_frames = np.concatenate([
        np.expand_dims(frames[0], axis=0),
        np.expand_dims(frames[1], axis=0),
        stddev_frames,
        np.expand_dims(frames[-2], axis=0),
        np.expand_dims(frames[-1], axis=0)
    ], axis=0)

    assert len(stddev_frames) == len(frames)
    output_dir = os.path.join(args.output_dir, args.domain, video_name)
    # output_dir = 'test'
    os.makedirs(output_dir, exist_ok=True)
    for i in range(len(stddev_frames)):
        Image.fromarray(stddev_frames[i].astype(np.uint8), mode='L').save(os.path.join(output_dir, f'{i}.jpg'), quality=95)

def save_sum_minus_5b_frames(frames, args, video_name):
    # calculate the global background
    # extend dimension of all frames to 3
    # apply a gaussian blur to the frames
    frames = np.array(frames)
    print(f'frames shape: {frames.shape}')
    # Process each frame individually with Gaussian blur
    blurred_frames = []
    for i in range(frames.shape[0]):  # Iterate over the time dimension
        frame = frames[i]  # Get single frame of shape (837, 628)
        blurred_frame = cv2.GaussianBlur(frame.astype(np.uint8), (5, 5), 0)
        blurred_frames.append(blurred_frame)
    frames = np.array(blurred_frames)
    
    print(f'blurred frames shape: {frames.shape}')
    # print(f'frames shape: {frames.shape}')
    global_background = np.mean(frames, axis=0)
    # print(f'global_background shape: {global_background.shape}')
    # get 5 frames at a time centered on the current frame, with a stride of 1
    sum_minus_5b_frames = []

    for i in range(2, len(frames) - 2):
        summed_frame = np.sum(frames[i-2:i+3], axis=0)
        sf = summed_frame - 5 * global_background
        # take only the positive values
        sf = np.where(sf > 0, sf, np.zeros_like(sf))
        sum_minus_5b_frames.append(sf)
    # add the raw frames to the front of the list and the back of the list
    sum_minus_5b_frames = np.array(sum_minus_5b_frames)
    sum_minus_5b_frames = np.concatenate([
        np.expand_dims(frames[0] - global_background, axis=0),
        np.expand_dims(frames[1] - global_background, axis=0),
        sum_minus_5b_frames,
        np.expand_dims(frames[-2] - global_background, axis=0),
        np.expand_dims(frames[-1] - global_background, axis=0)
    ], axis=0)

    assert len(sum_minus_5b_frames) == len(frames)
    output_dir = os.path.join(args.output_dir, args.domain, video_name)
    # output_dir = 'test'
    os.makedirs(output_dir, exist_ok=True)
    for i in range(len(sum_minus_5b_frames)):
        Image.fromarray(sum_minus_5b_frames[i].astype(np.uint8), mode='L').save(os.path.join(output_dir, f'{i}.jpg'), quality=95)


def save_sum_minus_3b_frames(frames, args, video_name):
    # calculate the global background
    # extend dimension of all frames to 3
    # apply a gaussian blur to the frames
    frames = np.array(frames)
    # Process each frame individually with Gaussian blur
    blurred_frames = []
    for i in range(frames.shape[0]):  # Iterate over the time dimension
        frame = frames[i]  # Get single frame of shape (837, 628)
        blurred_frame = cv2.GaussianBlur(frame.astype(np.uint8), (5, 5), 0)
        blurred_frames.append(blurred_frame)
    frames = np.array(blurred_frames)
    
    # print(f'frames shape: {frames.shape}')
    global_background = np.mean(frames, axis=0)
    # print(f'global_background shape: {global_background.shape}')
    # get 3 frames at a time centered on the current frame, with a stride of 1
    sum_minus_3b_frames = []

    for i in range(1, len(frames) - 1):
        summed_frame = np.sum(frames[i-1:i+2], axis=0)
        sf = summed_frame - 3 * global_background
        # take only the positive values
        sf = np.where(sf > 0, sf, np.zeros_like(sf))
        sum_minus_3b_frames.append(sf)
    # add the raw frames to the front of the list and the back of the list
    sum_minus_3b_frames = np.array(sum_minus_3b_frames)
    sum_minus_3b_frames = np.concatenate([
        np.expand_dims(frames[0] - global_background, axis=0),
        sum_minus_3b_frames,
        np.expand_dims(frames[-1] - global_background, axis=0)
    ], axis=0)

    assert len(sum_minus_3b_frames) == len(frames)
    output_dir = os.path.join(args.output_dir, args.domain, video_name)
    # output_dir = 'test'
    os.makedirs(output_dir, exist_ok=True)
    for i in range(len(sum_minus_3b_frames)):
        Image.fromarray(sum_minus_3b_frames[i].astype(np.uint8), mode='L').save(os.path.join(output_dir, f'{i}.jpg'), quality=95)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--frames_dir', type=str, required=True, help='Path to the raw frames')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save the target frames')
    parser.add_argument('--domain', type=str, required=True, help='Domain to process')
    parser.add_argument('--target', type=str, required=True, help='Target type')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # args.domain = 'kenai-channel'
    
    print(f'Processing domain {args.domain}')
    all_videos = [f for f in os.listdir(os.path.join(args.frames_dir, args.domain)) if os.path.isdir(os.path.join(args.frames_dir, args.domain, f)) and not f.startswith('.')]
    # all_videos = ["2018-08-16-JD228_Channel_Stratum1_Set1_CH_2018-08-16_060006_532_732"]
    for video in tqdm(all_videos):
        print(f'Processing video {video}')
        raw_video_path = os.path.join(args.frames_dir, args.domain, video)
        new_video_path = os.path.join(args.output_dir, args.domain, video)
        if os.path.exists(new_video_path) and len(os.listdir(new_video_path)) == len(os.listdir(raw_video_path)):
            print(f'Skipping {video} because it already exists at {new_video_path}')
            continue

        # Save the combined image to the folder location 
        os.makedirs(new_video_path, exist_ok=True)
            
        # Get all of the frames in the video
        raw_frame_names = sorted(os.listdir(raw_video_path), key=lambda x: int(x.split('.')[0]))
        raw_frames_paths = [os.path.join(raw_video_path, f) for f in raw_frame_names]
        # Open the frames as images
        raw_frame_imgs = np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in raw_frames_paths])            
        # Get the gap1 frames
        if args.target == "gap1_and_curr_frame":
            save_gap1_frames(raw_frame_imgs, args, video)
        elif args.target == "gap2_and_curr_frame":
            save_gap2_frames(raw_frame_imgs, args, video)
        elif args.target == 'stddev_5':
            save_stddev_5_frames(raw_frame_imgs, args, video)
        elif args.target == 'sum_minus_5b':
            save_sum_minus_5b_frames(raw_frame_imgs, args, video)
        elif args.target == 'sum_minus_3b':
            save_sum_minus_3b_frames(raw_frame_imgs, args, video)
        elif args.target == "identity":
            continue
        else:
            raise ValueError(f'Target type {args.target} not supported')
        