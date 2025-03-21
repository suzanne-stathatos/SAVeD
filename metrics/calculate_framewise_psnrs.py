# Given a domain, calculate the psnr for each frame within each clip of the domain
# Save the results to a csv file

# Input:
# - domain: the domain to calculate the psnr for
# - annotation_path: the path to the annotation file
# - pred_img_path: the path to the predicted image
# - output_path: the path to the output file

# Output:
# - csv file with the psnr for each frame within each clip of the domain

from argparse import ArgumentParser
from PIL import Image
import cv2
import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict
import torchvision.transforms as transforms
import torch
import numpy as np
from skimage import io, img_as_float
from psnr import PSNR_classical
from scipy.stats import wasserstein_distance

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--domain', type=str, default='kenai-train')
    parser.add_argument('--annotation_path', type=str, required=True)
    parser.add_argument('--target_type', type=str, default='gap1_and_curr_frame')
    parser.add_argument('--den_plus_img_path', type=str, required=True)
    parser.add_argument('--den_img_path', type=str, required=True)
    parser.add_argument('--raw_img_path', type=str, required=True)
    parser.add_argument('--bs_img_path', type=str, required=True)
    parser.add_argument('--bs_3_ch_img_path', type=str, required=True)
    parser.add_argument('--median_img_path', type=str, required=True)
    parser.add_argument('--gaussian_img_path', type=str, required=True)
    parser.add_argument('--n2v_img_path', type=str, required=True)
    parser.add_argument('--udvd_img_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    return args


def load_annotations(file_path, available_frames):
    """Load and verify annotation format"""
    annotations = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 7:  # Ensure we have all needed fields
                frame_idx = int(parts[0]) - 1 # Adjust for 0-based indexing
                assert frame_idx >=0, f"Frame index {frame_idx} is negative"
                if frame_idx in available_frames:
                    # Create bbox with x,y,w,h,conf
                    bbox = [float(x) for x in parts[2:7]]
                    # Make all boxes 25% larger
                    # x, y, w, h, conf = bbox
                    # x = x - w*0.25
                    # y = y - h*0.25
                    # w = w*1.5
                    # h = h*1.5
                    # bbox = [x, y, w, h, conf]
                    annotations[frame_idx].append(bbox)
    return annotations

transform = transforms.Compose([
    transforms.ToTensor(),
])

def load_target_image(path, clip_name, background, args):
    frame_num = int(path.split('.')[0].split('/')[-1])
    # Load frames
    frame1 = Image.open(path).convert('L')
    original_shape = frame1.size
    frame2 = Image.open(path.replace(f'{frame_num}.jpg', f'{frame_num-1}.jpg')).convert('L')
    frame3 = Image.open(path.replace(f'{frame_num}.jpg', f'{frame_num-2}.jpg')).convert('L')
    frame4 = Image.open(path.replace(f'{frame_num}.jpg', f'{frame_num+1}.jpg')).convert('L')
    frame5 = Image.open(path.replace(f'{frame_num}.jpg', f'{frame_num+2}.jpg')).convert('L')
    # convert frames to tensors
    frame1 = transform(frame1)
    transform_shape = frame1.shape
    frame2 = transform(frame2)
    frame3 = transform(frame3)
    frame4 = transform(frame4)
    frame5 = transform(frame5)
    if args.target_type == "l1":
        target = torch.abs(frame4 - frame5)
    elif args.target_type == "sum_minus_3_background":
        # Load global background if not already loaded
        background = Image.fromarray(background.astype(np.uint8))
        background = transform(background)
        target_non_abs = frame1 + frame2 + frame3 - 3*background
        target_positive = torch.where(target_non_abs > 0, target_non_abs, torch.zeros_like(target_non_abs))
        target = target_positive
    elif args.target_type == "sum_minus_5_background":
        # Load global background if not already loaded
        background = Image.fromarray(background.astype(np.uint8))
        background = transform(background)
        target_non_abs = frame1 + frame2 + frame3 + frame4 + frame5 - 5*background
        target_positive = torch.where(target_non_abs > 0, target_non_abs, torch.zeros_like(target_non_abs))
        target = target_positive
    elif args.target_type == "gap2_and_curr_frame":
        future_diff = frame5 - frame1 # exaggerates the motion in the forward direction
        future_diff = torch.where(future_diff > 0, future_diff, torch.zeros_like(future_diff))
        prev_diff = frame3 - frame1 # exaggerates the motion in the backward direction
        prev_diff = torch.where(prev_diff > 0, prev_diff, torch.zeros_like(prev_diff))
        target = frame1 + future_diff + prev_diff
    elif args.target_type == "gap1_and_curr_frame":
        future_diff = frame4 - frame1 # exaggerates the motion in the forward direction
        future_diff = torch.where(future_diff > 0, future_diff, torch.zeros_like(future_diff))
        prev_diff = frame2 - frame1 # exaggerates the motion in the backward direction
        prev_diff = torch.where(prev_diff > 0, prev_diff, torch.zeros_like(prev_diff))
        target = frame1 + future_diff + prev_diff
    elif args.target_type == "all_frame_diffs_and_curr":
        target = torch.abs(frame5 - frame4) + torch.abs(frame4 - frame3) + torch.abs(frame3 - frame2) + torch.abs(frame2 - frame1) + frame1
    elif args.target_type == "positive_diff_and_curr_frame":
        future_diff = frame5 - frame4 # exaggerates the motion in the forward direction
        future_diff = torch.where(future_diff > 0, future_diff, torch.zeros_like(future_diff))
        prev_diff = frame2 - frame1 # exaggerates the motion in the backward direction
        prev_diff = torch.where(prev_diff > 0, prev_diff, torch.zeros_like(prev_diff))
        target = frame1 + future_diff + prev_diff
    elif args.target_type == "identity":
        target = frame1
    elif args.target_type == "identity_t-1":
        target = frame2
    elif args.target_type == "l1_and_obj":
        target = torch.abs(frame4 - frame5)
    elif args.target_type == "bs_global":
        # Load global background if not already loaded
        background = Image.fromarray(background.astype(np.uint8))
        background = transform(background)
        target = frame1 - background
    elif args.target_type == "stddev_all":
        target = torch.std(torch.stack([frame3, frame2, frame1, frame4, frame5]), dim=0)
    elif args.target_type == "stddev_3":
        target = torch.std(torch.stack([frame2, frame1, frame4]), dim=0)
    else:
        raise ValueError(f"Unknown target type: {args.target_type}")
    return target


def calculate_background(clip_path):
    # get all frames for a clip
    full_clip = os.path.join(clip_path)  
    all_frames = [f for f in os.listdir(full_clip) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]
    frames = np.stack([Image.open(os.path.join(full_clip, f)).convert('L') for f in all_frames])
    blurred_frames = frames.astype(np.float32)
    # Blur frames
    for i in range(frames.shape[0]):
        blurred_frames[i] = cv2.GaussianBlur(blurred_frames[i], (5,5), 0)
    return np.mean(blurred_frames, axis=0)


def main(args):
    domain = args.domain
    annotation_path = args.annotation_path
    pred_img_path = args.pred_img_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    output_csv_path = os.path.join(output_path, 'framewise_psnrs.csv')
    # pandas dataframe to store the results
    results_df = pd.DataFrame(columns=['clip', 'frame', 'psnr_bw', 'psnr_target'])

    # iterate of the clips in the domain, getting the bounding boxes for each clip and frame
    for clip in os.listdir(os.path.join(args.pred_img_path, domain)):
        clip_path = os.path.join(args.pred_img_path, domain, clip)
        assert os.path.isdir(clip_path), f'Clip {clip} is not a directory'
        # load the annotation file for the clip
        available_frames = set(int(f.split('.')[0]) for f in os.listdir(clip_path) if f.endswith('.jpg'))
        raw_available_frames = set(int(f.split('.')[0]) for f in os.listdir(os.path.join(args.raw_img_path, domain, clip)) if f.endswith('.jpg'))
        annotations = load_annotations(os.path.join(annotation_path, domain, clip, 'gt.txt'), available_frames)
        if args.target_type == "bs_global" or args.target_type == "sum_minus_3_background" or args.target_type == "sum_minus_5_background":
            # load the global background for the clip
            background = calculate_background(clip_path)
        else:
            background = None

        for frame in tqdm(os.listdir(clip_path), desc=f'Calculating psnr for clip {clip}'):
            # check that frame+1, frame+2, frame-1, frame-2 exist
            frame_num = int(frame.split('.')[0])
            raw_frame_num = frame_num + 2
            if not (raw_frame_num in raw_available_frames and raw_frame_num+1 in raw_available_frames and raw_frame_num-1 in raw_available_frames and raw_frame_num-2 in raw_available_frames and raw_frame_num+2 in raw_available_frames):
                print(f"Skipping frame {frame} because it does not have the required neighboring frames")
                continue

            bboxes = annotations[frame]
            # load the target image for the frame
            frame_path = os.path.join(clip_path, frame)
            # print(f"frame_path: {frame_path}")
            raw_frame_path = os.path.join(args.raw_img_path, domain, clip, f"{raw_frame_num}.jpg")
            target_img = load_target_image(raw_frame_path, clip, background, args)    
            psnr = PSNR_classical(frame_path, bboxes, clip, frame)
            psnr_bw = psnr.calculate_psnr_image_vs_bw()
            psnr_target = psnr.calculate_psnr_image_vs_target(target_img)
            results_df = results_df._append({'clip': clip, 'frame': frame, 'psnr_bw': psnr_bw, 'psnr_target': psnr_target}, ignore_index=True)
    results_df.to_csv(output_csv_path, index=False)

    # Calculate which clip has the highest average psnr by each metric
    clip_averages_bw = {}
    clip_averages_target = {}
    for clip in results_df['clip'].unique():
        clip_averages_bw[clip] = results_df[results_df['clip'] == clip]['psnr_bw'].mean()
        clip_averages_target[clip] = results_df[results_df['clip'] == clip]['psnr_target'].mean()
    # print out which clip has the highest average psnr by each metric
    print(f"Clip with highest average psnr by bw: {max(clip_averages_bw, key=clip_averages_bw.get)}, psnr: {clip_averages_bw[max(clip_averages_bw, key=clip_averages_bw.get)]}")
    print(f"Clip with highest average psnr by target: {max(clip_averages_target, key=clip_averages_target.get)}, psnr: {clip_averages_target[max(clip_averages_target, key=clip_averages_target.get)]}")


import numpy as np
from scipy.stats import skew, kurtosis
import random
import matplotlib.pyplot as plt

def bimodality_coefficient(data):
  """
  Calculates Sarle's bimodality coefficient for a given dataset.

  Args:
    data: A 1D numpy array or list of numerical data.

  Returns:
    The bimodality coefficient (float) or None if the calculation is not possible.
  """
  # Ensure data is a numpy array
  data = np.array(data)

  # Handle edge cases where calculation is not possible
  if len(data) < 4 or np.std(data) == 0:
    return None

  # Calculate skewness and kurtosis
  s = skew(data)
  k = kurtosis(data)

  # Calculate the bimodality coefficient
  b = (s**2 + 1) / (k + (3*(len(data)-1)**2)/((len(data)-2)*(len(data)-3)))

  return b


def pixels_in_boxes(bboxes, frame):
    for bbox in bboxes:
        x1, y1, w, h, conf = bbox
        # convert to int
        x1, y1, w, h = int(x1), int(y1), int(w), int(h)
        frame[y1:y1+h, x1:x1+w] = 1
    return frame


def visualize_bimodality_coefficient_on_frames(args, clips):
    for clip in clips:
        clip_path = os.path.join(args.pred_img_path, args.domain, clip)
        available_frames = set(int(f.split('.')[0]) for f in os.listdir(clip_path) if f.endswith('.jpg'))
        raw_available_frames = set(int(f.split('.')[0]) for f in os.listdir(os.path.join(args.raw_img_path, args.domain, clip)) if f.endswith('.jpg'))
        annotations = load_annotations(os.path.join(args.annotation_path, args.domain, clip, 'gt.txt'), available_frames)
        # print(f"annotations: {annotations}")
        bimodality_coefficients = []
        # load all frames in the clip
        frames = [Image.open(os.path.join(clip_path, f)) for f in os.listdir(clip_path) if f.endswith('.jpg')]
        find_cone = np.max(frames, axis=0)
        unique_values, counts = np.unique(find_cone, return_counts=True)
        # Find the most frequent value
        most_frequent_value = unique_values[np.argmax(counts)]
        print(f"most_frequent_value: {most_frequent_value}")

        find_cone = find_cone[:,:,0]
        cone_indices = np.where(find_cone == most_frequent_value ) # assuming the cone is black
        non_cone_indices = np.where(find_cone != most_frequent_value)
        non_cone_mask = np.zeros_like(find_cone)
        non_cone_mask[find_cone != most_frequent_value] = 1

        for frame_id, key in enumerate(annotations.keys()):
            if frame_id > 3:
                break
            bboxes = annotations[key]
            print(f"bboxes: {bboxes}")
            assert len(bboxes) > 0, f"No bboxes found for frame {key}"
            loaded_frame = frames[key]
            loaded_frame = np.array(loaded_frame)[:,:,0]
            # take the pixels of the loaded_frame that are not in the cone_indices
            pixels_not_cone = loaded_frame[non_cone_indices]
            zeros = np.zeros_like(loaded_frame)
            fish_pixels_mask = pixels_in_boxes(bboxes, zeros)
            print(np.unique(fish_pixels_mask))
            fish_pixels = loaded_frame * fish_pixels_mask
            print('fish_pixels', fish_pixels.shape)

            fish_pixels_cutout = fish_pixels[fish_pixels_mask == 1]
            print('fish_pixels_cutout', fish_pixels_cutout.shape)

            non_fish_pixels = loaded_frame 
            non_fish_mask = (1-fish_pixels_mask) * non_cone_mask
            print('non_fish_mask', non_fish_mask.shape)
            non_fish_pixels_cutout = non_fish_pixels[non_fish_mask == 1]
            print('non_fish_pixels_cutout', non_fish_pixels_cutout.shape)
            non_fish_pixels[non_fish_mask] = 0
            print('non_fish_pixels', non_fish_pixels.shape)

            fish_bimodality = bimodality_coefficient(fish_pixels_cutout.flatten())
            brnd_bimodality = bimodality_coefficient(non_fish_pixels_cutout.flatten())

            print(f"fish_bimodality: {fish_bimodality}, brnd_bimodality: {brnd_bimodality}")
            # non_fish_pixels[non_cone_mask] = -1

            fig, ax = plt.subplots(2,3)
            ax[0,0].imshow(fish_pixels)
            ax[0,1].imshow(non_fish_pixels)
            ax[0,2].imshow(non_cone_mask)
            ax[1,0].hist(fish_pixels_cutout.flatten(), bins=50)
            ax[1,1].hist(non_fish_pixels_cutout.flatten(), bins=50)
            ax[1,2].hist(non_fish_pixels_cutout.flatten(), bins=50, alpha=0.5, density=True)
            ax[1,2].hist(fish_pixels_cutout.flatten(), bins=50, alpha=0.5, density=True)
            plt.savefig(f"bimodality_{clip}_{frame_id}.png")


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IOU) between two bounding boxes.
    Each box should be in format [x, y, width, height]
    """
    try:
        x1, y1, w1, h1 = map(float, box1[:4])  # Convert to float to avoid integer division
        x2, y2, w2, h2 = map(float, box2[:4])
        
        # Ensure positive dimensions
        if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
            return 0.0
        
        # Calculate intersection coordinates
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        
        intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        if union <= 0:  # Avoid division by zero
            return 0.0
            
        iou = intersection / union
        
        # Sanity check on IOU value
        if not (0 <= iou <= 1):
            print(f"Warning: Invalid IOU value {iou}")
            print(f"Box1: {box1}")
            print(f"Box2: {box2}")
            return 0.0
            
        return iou
        
    except Exception as e:
        print(f"Error in IOU calculation: {e}")
        print(f"Box1: {box1}")
        print(f"Box2: {box2}")
        return 0.0


def iou_histogram(img1, img2):
    img1 = img1.copy()/255.
    img2 = img2.copy()/255.
    h1, bins1 = np.histogram(img1.flatten(), bins=255, range=(0., 1.), density=False)
    h2, bins2 = np.histogram(img2.flatten(), bins=255, range=(0., 1.), density=False)
    # print(bins1[:5])
    # print(bins2[:5])
    intersection = np.sum(np.minimum(h1, h2))
    # plt.subplot(2, 3, 5)
    # plt.scatter(bins1[:-1], np.minimum(h1, h2))
    union = np.sum(h1) + np.sum(h2) - intersection
    assert np.sum(h1) == np.sum(h2), f"h1 and h2 are not the same: {np.sum(h1)} != {np.sum(h2)}"
    return intersection / union


def hellinger_distance(img1, img2):
    img1_h = img1.flatten().copy() / 255.   #/np.sum(img1.flatten())
    img2_h = img2.flatten().copy() / 255.   #/np.sum(img2.flatten())
    h1, bins1 = np.histogram(img1_h, bins=255, range=(0., 1.), density=False)
    h2, bins2 = np.histogram(img2_h, bins=255, range=(0., 1.), density=False)
    h1 = h1/np.sum(h1)
    h2 = h2/np.sum(h2)
    return np.sqrt(1 - np.sum(np.sqrt(h1 * h2)))

def kl_divergence(img1, img2):
    img1_h = img1.copy() / 255.0
    img2_h = img2.copy() / 255.0

    # Compute histograms
    h1, bins1 = np.histogram(img1_h.flatten(), bins=255, range=(0., 1.), density=True)
    h2, bins2 = np.histogram(img2_h.flatten(), bins=255, range=(0., 1.), density=True)
    p, q = h1, h2
    # Avoid division by zero or log(0) by adding a small epsilon
    epsilon = 1e-10
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)
    return np.sum(p * np.log(p / q))

def chi_squared_distance(img1, img2):
    img1 = img1.copy() / 255.0
    img2 = img2.copy() / 255.0
    p, bins1 = np.histogram(img1.flatten(), bins=255, range=(0., 1.), density=True)
    q, bins2 = np.histogram(img2.flatten(), bins=255, range=(0., 1.), density=True)
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-10
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)
    return np.sum((p - q)**2 / (p + q))


def wasserstein_dist(img1, img2):
    img1_h = img1.copy()/255.
    img2_h = img2.copy()/255.
    h1, bins1 = np.histogram(img1_h.flatten(), bins=255, range=(0., 1.), density=True)
    h2, bins2 = np.histogram(img2_h.flatten(), bins=255, range=(0., 1.), density=True)
    # Calculate Wasserstein distance
    return wasserstein_distance(bins1[:-1], bins2[:-1], h1, h2)


def visualize_psnr_on_frames_fish_boxes_vs_frames_no_fish(args, clips):
    for clip in clips:
        den_clip_path = os.path.join(args.den_img_path, args.domain, clip)
        raw_clip_path = os.path.join(args.raw_img_path, args.domain, clip)
        bs_clip_path = os.path.join(args.bs_img_path, args.domain, clip)
        bs_3_ch_clip_path = os.path.join(args.bs_3_ch_img_path, args.domain, clip)
        den_plus_clip_path = os.path.join(args.den_plus_img_path, args.domain, clip)
        
        available_frames = set(int(f.split('.')[0]) for f in os.listdir(den_clip_path) if f.endswith('.jpg'))
        annotations = load_annotations(os.path.join(args.annotation_path, args.domain, clip, 'gt.txt'), available_frames)
        # load all frames in the clip
        all_frames = sorted(os.listdir(den_clip_path), key=lambda x: int(x.split('.')[0]))
        denoised_frames = [(os.path.join(den_clip_path, f), f) for f in all_frames if f.endswith('.jpg')]
        raw_frames = [(os.path.join(raw_clip_path, f), f) for f in all_frames if f.endswith('.jpg')]
        bs_frames = [(os.path.join(bs_clip_path, f), f) for f in all_frames if f.endswith('.jpg')]
        bs_3_ch_frames = [(os.path.join(bs_3_ch_clip_path, f), f) for f in all_frames if f.endswith('.jpg')]
        den_plus_frames = [(os.path.join(den_plus_clip_path, f), f) for f in all_frames if f.endswith('.jpg')]
        for _, frame_idx in enumerate(annotations.keys()):
            frame_num = raw_frames[frame_idx][1].split('.')[0]
            if frame_num != '126':
            # if frame_num != '153':
                continue
            copy_annos = annotations.copy()
            bboxes = copy_annos[frame_idx]
            den_frame = np.array(Image.open(denoised_frames[frame_idx][0]).convert('RGB'))
            raw_frame = np.array(Image.open(raw_frames[frame_idx][0]).convert('RGB'))
            den_plus_frame = np.array(Image.open(den_plus_frames[frame_idx][0]).convert('RGB'))
            bs_frame = np.array(Image.open(bs_frames[frame_idx][0]).convert('RGB'))
            bs_3_ch_frame = np.array(Image.open(bs_3_ch_frames[frame_idx][0]).convert('RGB'))
            den_frame_with_bbox_anno = den_frame.copy()
            raw_frame_with_bbox_anno = raw_frame.copy()
            den_plus_frame_with_bbox_anno = den_plus_frame.copy()
            bs_frame_with_bbox_anno = bs_frame.copy()
            bs_3_ch_frame_with_bbox_anno = bs_3_ch_frame.copy()
            for bbox in bboxes:
                x, y, w, h, conf = bbox
                # convert to int
                x, y, w, h = int(x), int(y), int(w), int(h)
                # find same bbox location of a frame without any fish 
                empty_box_found = False
                for k, (other_den_frame, other_den_frame_name) in enumerate(denoised_frames):
                    if k == frame_idx:
                        continue
                    if empty_box_found:
                        continue
                    other_bboxes = copy_annos[k]
                    for other_bbox in other_bboxes:
                        if calculate_iou(other_bbox, bbox) == 0:
                            # print(f"No fish found in other frame {other_den_frame_name} at same location {bbox}")
                            other_den_frame_raw = np.array(Image.open(raw_frames[k][0]).convert('RGB'))
                            other_den_frame = np.array(Image.open(denoised_frames[k][0]).convert('RGB'))
                            other_bs_frame = np.array(Image.open(bs_frames[k][0]).convert('RGB'))
                            other_bs_3_ch_frame = np.array(Image.open(bs_3_ch_frames[k][0]).convert('RGB'))
                            other_den_plus_frame = np.array(Image.open(den_plus_frames[k][0]).convert('RGB'))

                            other_den_frame_bbox = other_den_frame[y:y+h, x:x+w, :].copy()
                            den_frame_bbox = den_frame[y:y+h, x:x+w, :].copy()
                            raw_frame_bbox = raw_frame[y:y+h, x:x+w, :].copy()
                            other_raw_frame_bbox = other_den_frame_raw[y:y+h, x:x+w, :].copy()
                            bs_frame_bbox = bs_frame[y:y+h, x:x+w, :].copy()
                            other_bs_frame_bbox = other_bs_frame[y:y+h, x:x+w, :].copy()
                            bs_3_ch_frame_bbox = bs_3_ch_frame[y:y+h, x:x+w, :].copy()
                            other_bs_3_ch_frame_bbox = other_bs_3_ch_frame[y:y+h, x:x+w, :].copy()
                            den_plus_frame_bbox = den_plus_frame[y:y+h, x:x+w, :].copy()
                            other_den_plus_frame_bbox = other_den_plus_frame[y:y+h, x:x+w, :].copy()

                            other_den_frame_with_bbox_anno = other_den_frame.copy()
                            other_raw_frame_with_bbox_anno = other_den_frame_raw.copy()
                            other_bs_3_ch_frame_with_bbox_anno = other_bs_3_ch_frame#.copy()
                            other_bs_frame_with_bbox_anno = other_bs_frame.copy()
                            other_den_plus_frame_with_bbox_anno = other_den_plus_frame#.copy()

                            other_den_frame_bbox = other_den_frame[y:y+h, x:x+w, :].copy()

                            den_frame_bbox = den_frame[y:y+h, x:x+w, :].copy()
                            other_den_frame_raw_bbox = other_den_frame_raw[y:y+h, x:x+w, :].copy()

                            raw_frame_bbox = raw_frame[y:y+h, x:x+w, :].copy()
                            other_raw_frame_bbox = other_den_frame_raw[y:y+h, x:x+w, :].copy()

                            bs_frame_bbox = bs_frame[y:y+h, x:x+w, :].copy()
                            other_bs_frame_bbox = other_bs_frame[y:y+h, x:x+w, :].copy()

                            bs_3_ch_frame_bbox = bs_3_ch_frame[y:y+h, x:x+w, :].copy()
                            other_bs_3_ch_frame_bbox = other_bs_3_ch_frame[y:y+h, x:x+w, :].copy()

                            den_plus_frame_bbox = den_plus_frame[y:y+h, x:x+w, :].copy()
                            other_den_plus_frame_bbox = other_den_plus_frame[y:y+h, x:x+w, :].copy()

                            # calculate metrics
                            iou_raw = iou_histogram(raw_frame_bbox, other_raw_frame_bbox)
                            kl_raw = kl_divergence(other_raw_frame_bbox, raw_frame_bbox)
                            hellinger_raw = hellinger_distance(other_raw_frame_bbox, raw_frame_bbox)
                            chi2_raw = chi_squared_distance(other_raw_frame_bbox, raw_frame_bbox)
                            wasserstein_raw = wasserstein_dist(raw_frame_bbox, other_raw_frame_bbox)
                            print(f"frame {frame_idx} | raw | iou: {iou_raw:.4f} | hellinger: {hellinger_raw:.4f}  | kl: {kl_raw:.4f} | chi2: {chi2_raw:.4f} | wasserstein: {wasserstein_raw:.4f}")
                            iou_bs = iou_histogram(bs_frame_bbox, other_bs_frame_bbox)
                            kl_bs = kl_divergence(other_bs_frame_bbox, bs_frame_bbox)
                            kl_bs_3_ch = kl_divergence(other_bs_3_ch_frame_bbox, bs_3_ch_frame_bbox)
                            hellinger_bs = hellinger_distance(bs_frame_bbox, other_bs_frame_bbox)
                            chi2_bs = chi_squared_distance(bs_frame_bbox, other_bs_frame_bbox)
                            wasserstein_bs = wasserstein_dist(other_bs_frame_bbox, bs_frame_bbox)
                            print(f"frame {frame_idx} | bs | iou: {iou_bs:.4f} | hellinger: {hellinger_bs:.4f} | kl: {kl_bs:.4f} | chi2: {chi2_bs:.4f} | wasserstein: {wasserstein_bs:.4f}")
                            iou_den = iou_histogram(den_frame_bbox, other_den_frame_bbox)
                            kl_den = kl_divergence(other_den_frame_bbox, den_frame_bbox)
                            hellinger_den = hellinger_distance(den_frame_bbox, other_den_frame_bbox)
                            chi2_den = chi_squared_distance(den_frame_bbox, other_den_frame_bbox)
                            wasserstein_den = wasserstein_dist(other_den_frame_bbox, den_frame_bbox)
                            print(f"frame {frame_idx} | den | iou: {iou_den:.4f} | hellinger: {hellinger_den:.4f} | kl: {kl_den:.4f} | chi2: {chi2_den:.4f} | wasserstein: {wasserstein_den:.4f}")
                            iou_den_plus = iou_histogram(den_plus_frame_bbox, other_den_plus_frame_bbox)
                            kl_den_plus = kl_divergence(other_den_plus_frame_bbox, den_plus_frame_bbox)
                            hellinger_den_plus = hellinger_distance(den_plus_frame_bbox, other_den_plus_frame_bbox)
                            chi2_den_plus = chi_squared_distance(den_plus_frame_bbox, other_den_plus_frame_bbox)
                            wasserstein_den_plus = wasserstein_dist(other_den_plus_frame_bbox, den_plus_frame_bbox)
                            if x == 464 and y == 260:
                                print(x,y)
                                # do not annotate
                                continue

                            # show box in red on the pred_frame
                            den_frame_with_bbox_anno = cv2.rectangle(den_frame_with_bbox_anno.copy(), (x, y), (x+w, y+h), (255, 0, 0), 2)
                            other_den_frame_with_bbox_anno = cv2.rectangle(other_den_frame_with_bbox_anno.copy(), (x, y), (x+w, y+h), (255, 0, 0), 2)
                            raw_frame_with_bbox_anno = cv2.rectangle(raw_frame_with_bbox_anno.copy(), (x, y), (x+w, y+h), (255, 0, 0), 2)
                            other_raw_frame_with_bbox_anno = cv2.rectangle(other_raw_frame_with_bbox_anno.copy(), (x, y), (x+w, y+h), (255, 0, 0), 2)
                            bs_frame_with_bbox_anno = cv2.rectangle(bs_frame_with_bbox_anno.copy(), (x, y), (x+w, y+h), (255, 0, 0), 2)
                            bs_3_ch_frame_with_bbox_anno = cv2.rectangle(bs_3_ch_frame_with_bbox_anno.copy(), (x, y), (x+w, y+h), (255, 0, 0), 2)
                            other_bs_3_ch_frame_with_bbox_anno = cv2.rectangle(other_bs_3_ch_frame_with_bbox_anno, (x, y), (x+w, y+h), (255, 0, 0), 2)
                            # annotate the KL divergence on the bs_3_ch_frame_with_bbox_anno
                            cv2.putText(bs_3_ch_frame_with_bbox_anno, f"KL:{kl_bs_3_ch:.3g}", (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                            den_plus_frame_with_bbox_anno = cv2.rectangle(den_plus_frame_with_bbox_anno.copy(), (x, y), (x+w, y+h), (255, 0, 0), 2)
                            other_den_plus_frame_with_bbox_anno = cv2.rectangle(other_den_plus_frame_with_bbox_anno, (x, y), (x+w, y+h), (255, 0, 0), 1)
                            # annotate the KL divergence on the den_plus_frame_with_bbox_anno
                            cv2.putText(den_plus_frame_with_bbox_anno, f"KL:{kl_den_plus:.3g}", (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                            # Also annotate the KL divergence on the frame

                            empty_box_found = True
                            fig, ax = plt.subplots(1,2, figsize=(9,9))
                            # ax[0].imshow(other_bs_3_ch_frame_with_bbox_anno)
                            # ax[0].set_title(f"Baseline 3ch NO FISH frame {k}")
                            ax[0].imshow(bs_3_ch_frame_with_bbox_anno)
                            ax[0].set_title(f"Baseline 3ch FISH frame {frame_idx}")
                            ax[0].axis('off')
                            # ax[2].imshow(other_den_plus_frame_with_bbox_anno)
                            # ax[2].set_title(f"Denoised+ NO FISH frame {k}")
                            ax[1].imshow(den_plus_frame_with_bbox_anno)
                            ax[1].set_title(f"Denoised+ FISH frame {frame_idx}")
                            ax[1].axis('off')
                            # ax[3].imshow(den_frame_with_bbox_anno)
                            # ax[0,0].imshow(other_raw_frame_with_bbox_anno)
                            # ax[0,0].set_title(f"Raw NO FISH frame {k}")
                            # ax[1,0].hist(raw_frame_bbox.flatten(), bins=50, alpha=0.5, density=False, label='FISH')
                            # ax[1,0].hist(other_raw_frame_bbox.flatten(), bins=50, alpha=0.5, density=False, label='NO FISH')
                            # ax[1,0].set_title(f"Raw histogram of\n pixel intensities (PI)")
                            # ax[1,0].legend()
                            # ax[0,1].imshow(bs_frame_with_bbox_anno)
                            # ax[0,1].set_title(f"BkgrndSub FISH frame {frame_idx}")
                            # ax[1,1].hist(bs_frame_bbox.flatten(), bins=50, alpha=0.5, density=False, label='FISH')
                            # ax[1,1].hist(other_bs_frame_bbox.flatten(), bins=50, alpha=0.5, density=False, label='NO FISH')
                            # ax[1,1].set_title(f"BkgrndSub histogram of PI")
                            # ax[1,1].legend()
                            # ax[0,2].imshow(den_frame_with_bbox_anno)
                            # ax[0,2].set_title(f"Denoised FISH frame {frame_idx}")
                            # ax[1,2].hist(den_frame_bbox.flatten(), bins=50, alpha=0.5, density=False, label='FISH')
                            # ax[1,2].hist(other_den_frame_bbox.flatten(), bins=50, alpha=0.5, density=False, label='NO FISH')
                            # ax[1,2].set_title(f"Denoised histogram of PI")
                            # ax[1,2].legend()
                            # ax[0,3].imshow(den_plus_frame_with_bbox_anno)
                            # ax[0,3].set_title(f"Denoised+ FISH frame {frame_idx}")
                            # ax[1,3].hist(den_plus_frame_bbox.flatten(), bins=50, alpha=0.5, density=False, label='FISH')
                            # ax[1,3].hist(other_den_plus_frame_bbox.flatten(), bins=50, alpha=0.5, density=False, label='NO FISH')
                            # ax[1,3].set_title(f"Denoised+ histogram of PI")
                            # ax[1,3].legend()

                            # save figure
                            print(f"frame {frame_idx} | den_plus | iou: {iou_den_plus:.4f} | hellinger: {hellinger_den_plus:.4f} | kl: {kl_den_plus:.4f} | chi2: {chi2_den_plus:.4f} | wasserstein: {wasserstein_den_plus:.4f}")
                            os.makedirs(f"metrics/bbox_comparison_{args.domain}", exist_ok=True)
                            plt.savefig(f"metrics/bbox_comparison_{args.domain}/{clip}_{frame_idx}.png")
                            print(f"Saved figure to metrics/bbox_comparison_{args.domain}/{clip}_{frame_idx}.png")

                            continue
                    if not empty_box_found:
                        print(f"No empty box found for frame {frame_idx} at location {bbox}")
                        pass


def calculate_metrics_on_frames(args, clips):
    clip_with_highest_kl_div = None
    frame_with_highest_kl_div = None
    highest_kl_div = 0 
    highest_kl_div_diff = 0

    # create dictionaries for each metric
    raw_hellingers_clipwise = {}
    bs_hellingers_clipwise = {}
    bs_3_ch_hellingers_clipwise = {}
    den_hellingers_clipwise = {}
    den_plus_hellingers_clipwise = {}

    raw_iou_clipwise = {}
    bs_iou_clipwise = {}
    bs_3_ch_iou_clipwise = {}
    den_iou_clipwise = {}
    den_plus_iou_clipwise = {}

    raw_kl_clipwise = {}
    bs_kl_clipwise = {}
    bs_3_ch_kl_clipwise = {}
    den_kl_clipwise = {}
    den_plus_kl_clipwise = {}
    median_kl_clipwise = {}
    gaussian_kl_clipwise = {}
    n2v_kl_clipwise = {}
    udvd_kl_clipwise = {}

    raw_chi2_clipwise = {}
    bs_chi2_clipwise = {}
    bs_3_ch_chi2_clipwise = {}
    den_chi2_clipwise = {}
    den_plus_chi2_clipwise = {}

    raw_wasserstein_clipwise = {}
    bs_wasserstein_clipwise = {}
    bs_3_ch_wasserstein_clipwise = {}
    den_wasserstein_clipwise = {}
    den_plus_wasserstein_clipwise = {}

    # create lists for total metrics
    count_total_frames = 0
    raw_total_hellingers = 0
    raw_total_iou = 0
    raw_total_kl = 0
    raw_total_chi2 = 0
    raw_total_wasserstein = 0
    bs_total_hellingers = 0
    bs_total_iou = 0
    bs_total_kl = 0
    bs_total_chi2 = 0
    bs_total_wasserstein = 0
    bs_3_ch_total_hellingers = 0
    bs_3_ch_total_iou = 0
    bs_3_ch_total_kl = 0
    bs_3_ch_total_chi2 = 0
    bs_3_ch_total_wasserstein = 0
    den_total_hellingers = 0
    den_total_iou = 0
    den_total_kl = 0
    den_total_chi2 = 0
    den_total_wasserstein = 0
    den_plus_total_hellingers = 0
    den_plus_total_iou = 0
    den_plus_total_kl = 0
    den_plus_total_chi2 = 0
    den_plus_total_wasserstein = 0

    median_total_kl = 0
    gaussian_total_kl = 0
    n2v_total_kl = 0
    udvd_total_kl = 0

    for clip in tqdm(clips):
        raw_hellingers_clipwise[clip] = []
        bs_hellingers_clipwise[clip] = []
        bs_3_ch_hellingers_clipwise[clip] = []
        den_hellingers_clipwise[clip] = []
        den_plus_hellingers_clipwise[clip] = []

        raw_iou_clipwise[clip] = []
        bs_iou_clipwise[clip] = []
        bs_3_ch_iou_clipwise[clip] = []
        den_iou_clipwise[clip] = []
        den_plus_iou_clipwise[clip] = []

        raw_kl_clipwise[clip] = []
        bs_kl_clipwise[clip] = []
        bs_3_ch_kl_clipwise[clip] = []
        den_kl_clipwise[clip] = []
        den_plus_kl_clipwise[clip] = []

        raw_chi2_clipwise[clip] = []
        bs_chi2_clipwise[clip] = []
        bs_3_ch_chi2_clipwise[clip] = []
        den_chi2_clipwise[clip] = []
        den_plus_chi2_clipwise[clip] = []

        raw_wasserstein_clipwise[clip] = []
        bs_wasserstein_clipwise[clip] = []
        bs_3_ch_wasserstein_clipwise[clip] = []
        den_wasserstein_clipwise[clip] = []
        den_plus_wasserstein_clipwise[clip] = []

        median_kl_clipwise[clip] = []
        gaussian_kl_clipwise[clip] = []
        n2v_kl_clipwise[clip] = []
        udvd_kl_clipwise[clip] = []

        raw_clip_path = os.path.join(args.raw_img_path, args.domain, clip)
        bs_clip_path = os.path.join(args.bs_img_path, args.domain, clip)
        bs_3_ch_clip_path = os.path.join(args.bs_3_ch_img_path, args.domain, clip)
        den_clip_path = os.path.join(args.den_img_path, args.domain, clip)
        den_plus_clip_path = os.path.join(args.den_plus_img_path, args.domain, clip)
        median_clip_path = os.path.join(args.median_img_path, args.domain, clip)
        gaussian_clip_path = os.path.join(args.gaussian_img_path, args.domain, clip)
        n2v_clip_path = os.path.join(args.n2v_img_path, args.domain, clip)
        udvd_clip_path = os.path.join(args.udvd_img_path, args.domain, clip)

        available_frames = set(int(f.split('.')[0]) for f in os.listdir(den_clip_path) if f.endswith('.jpg'))
        annotations = load_annotations(os.path.join(args.annotation_path, args.domain, clip, 'gt.txt'), available_frames)
        # load all frames in the clip
        all_frames = sorted(os.listdir(den_clip_path), key=lambda x: int(x.split('.')[0]))
        denoised_frames = [(os.path.join(den_clip_path, f), f) for f in all_frames if f.endswith('.jpg')]
        raw_frames = [(os.path.join(raw_clip_path, f), f) for f in all_frames if f.endswith('.jpg')]
        bs_frames = [(os.path.join(bs_clip_path, f), f) for f in all_frames if f.endswith('.jpg')]
        bs_3_ch_frames = [(os.path.join(bs_3_ch_clip_path, f), f) for f in all_frames if f.endswith('.jpg')]
        den_plus_frames = [(os.path.join(den_plus_clip_path, f), f) for f in all_frames if f.endswith('.jpg')]
        median_frames = [(os.path.join(median_clip_path, f), f) for f in all_frames if f.endswith('.jpg')]
        gaussian_frames = [(os.path.join(gaussian_clip_path, f), f) for f in all_frames if f.endswith('.jpg')]
        n2v_frames = [(os.path.join(n2v_clip_path, f), f) for f in all_frames if f.endswith('.jpg')]
        udvd_frames = [(os.path.join(udvd_clip_path, f), f) for f in all_frames if f.endswith('.jpg')]

        for _, frame_idx in enumerate(annotations.keys()):
            frame_num = raw_frames[frame_idx][1].split('.')[0]
            copy_annos = annotations.copy()
            bboxes = copy_annos[frame_idx]
            den_frame = np.array(Image.open(denoised_frames[frame_idx][0]).convert('RGB'))
            raw_frame = np.array(Image.open(raw_frames[frame_idx][0]).convert('RGB'))
            den_plus_frame = np.array(Image.open(den_plus_frames[frame_idx][0]).convert('RGB'))
            bs_frame = np.array(Image.open(bs_frames[frame_idx][0]).convert('RGB'))
            bs_3_ch_frame = np.array(Image.open(bs_3_ch_frames[frame_idx][0]).convert('RGB'))
            median_frame = np.array(Image.open(median_frames[frame_idx][0]).convert('RGB'))
            gaussian_frame = np.array(Image.open(gaussian_frames[frame_idx][0]).convert('RGB'))
            n2v_frame = np.array(Image.open(n2v_frames[frame_idx][0]).convert('RGB'))
            udvd_frame = np.array(Image.open(udvd_frames[frame_idx][0]).convert('RGB'))
            for bbox in bboxes:
                x, y, w, h, conf = bbox
                # convert to int
                x, y, w, h = int(x), int(y), int(w), int(h)
                # find same bbox location of a frame without any fish 
                empty_box_found = False
                for k, (other_den_frame, other_den_frame_name) in enumerate(denoised_frames):
                    if k == frame_idx:
                        continue
                    if empty_box_found:
                        continue
                    other_bboxes = copy_annos[k]
                    for other_bbox in other_bboxes:
                        if calculate_iou(other_bbox, bbox) == 0:
                            # print(f"No fish found in other frame {other_den_frame_name} at same location {bbox}")
                            other_den_frame_raw = np.array(Image.open(raw_frames[k][0]).convert('RGB'))
                            other_den_frame = np.array(Image.open(denoised_frames[k][0]).convert('RGB'))
                            other_bs_frame = np.array(Image.open(bs_frames[k][0]).convert('RGB'))
                            other_bs_3_ch_frame = np.array(Image.open(bs_3_ch_frames[k][0]).convert('RGB'))
                            other_den_plus_frame = np.array(Image.open(den_plus_frames[k][0]).convert('RGB'))
                            other_median_frame = np.array(Image.open(median_frames[k][0]).convert('RGB'))
                            other_gaussian_frame = np.array(Image.open(gaussian_frames[k][0]).convert('RGB'))
                            other_n2v_frame = np.array(Image.open(n2v_frames[k][0]).convert('RGB'))
                            other_udvd_frame = np.array(Image.open(udvd_frames[k][0]).convert('RGB'))

                            other_den_frame_bbox = other_den_frame[y:y+h, x:x+w, :].copy()
                            den_frame_bbox = den_frame[y:y+h, x:x+w, :].copy()
                            raw_frame_bbox = raw_frame[y:y+h, x:x+w, :].copy()
                            other_raw_frame_bbox = other_den_frame_raw[y:y+h, x:x+w, :].copy()
                            bs_frame_bbox = bs_frame[y:y+h, x:x+w, :].copy()
                            other_bs_frame_bbox = other_bs_frame[y:y+h, x:x+w, :].copy()
                            bs_3_ch_frame_bbox = bs_3_ch_frame[y:y+h, x:x+w, :].copy()
                            other_bs_3_ch_frame_bbox = other_bs_3_ch_frame[y:y+h, x:x+w, :].copy()
                            den_plus_frame_bbox = den_plus_frame[y:y+h, x:x+w, :].copy()
                            other_den_plus_frame_bbox = other_den_plus_frame[y:y+h, x:x+w, :].copy()
                            median_frame_bbox = median_frame[y:y+h, x:x+w, :].copy()
                            other_median_frame_bbox = other_median_frame[y:y+h, x:x+w, :].copy()
                            gaussian_frame_bbox = gaussian_frame[y:y+h, x:x+w, :].copy()
                            other_gaussian_frame_bbox = other_gaussian_frame[y:y+h, x:x+w, :].copy()
                            n2v_frame_bbox = n2v_frame[y:y+h, x:x+w, :].copy()
                            other_n2v_frame_bbox = other_n2v_frame[y:y+h, x:x+w, :].copy()
                            udvd_frame_bbox = udvd_frame[y:y+h, x:x+w, :].copy()
                            other_udvd_frame_bbox = other_udvd_frame[y:y+h, x:x+w, :].copy()

                            empty_box_found = True
                            # iou_raw = iou_histogram(raw_frame_bbox, other_raw_frame_bbox)
                            # hellinger_raw = hellinger_distance(other_raw_frame_bbox, raw_frame_bbox)
                            kl_raw = kl_divergence(raw_frame_bbox, other_raw_frame_bbox)
                            # chi2_raw = chi_squared_distance(raw_frame_bbox, other_raw_frame_bbox)
                            # wasserstein_raw = wasserstein_dist(raw_frame_bbox, other_raw_frame_bbox)
                            # iou_bs = iou_histogram(bs_frame_bbox, other_bs_frame_bbox)
                            # hellinger_bs = hellinger_distance(other_bs_frame_bbox, bs_frame_bbox)
                            kl_bs = kl_divergence(bs_frame_bbox, other_bs_frame_bbox)
                            # chi2_bs = chi_squared_distance(bs_frame_bbox, other_bs_frame_bbox)
                            # wasserstein_bs = wasserstein_dist(bs_frame_bbox, other_bs_frame_bbox)
                            # iou_bs_3_ch = iou_histogram(bs_3_ch_frame_bbox, other_bs_3_ch_frame_bbox)
                            # hellinger_bs_3_ch = hellinger_distance(other_bs_3_ch_frame_bbox, bs_3_ch_frame_bbox)
                            kl_bs_3_ch = kl_divergence(bs_3_ch_frame_bbox, other_bs_3_ch_frame_bbox)
                            # chi2_bs_3_ch = chi_squared_distance(bs_3_ch_frame_bbox, other_bs_3_ch_frame_bbox)
                            # wasserstein_bs_3_ch = wasserstein_dist(bs_3_ch_frame_bbox, other_bs_3_ch_frame_bbox)
                            # iou_den = iou_histogram(den_frame_bbox, other_den_frame_bbox)
                            # hellinger_den = hellinger_distance(other_den_frame_bbox, den_frame_bbox)
                            kl_den = kl_divergence(other_den_frame_bbox, den_frame_bbox)
                            # chi2_den = chi_squared_distance(other_den_frame_bbox, den_frame_bbox)
                            # wasserstein_den = wasserstein_dist(den_frame_bbox, other_den_frame_bbox)
                            # iou_den_plus = iou_histogram(den_plus_frame_bbox, other_den_plus_frame_bbox)
                            # hellinger_den_plus = hellinger_distance(other_den_plus_frame_bbox, den_plus_frame_bbox)
                            kl_den_plus = kl_divergence(den_plus_frame_bbox, other_den_plus_frame_bbox)
                            if kl_den_plus - kl_raw > highest_kl_div_diff:
                                highest_kl_div = kl_den_plus
                                highest_kl_div_diff = kl_den_plus - kl_raw
                                clip_with_highest_kl_div = clip
                                frame_with_highest_kl_div = frame_idx
                            # chi2_den_plus = chi_squared_distance(other_den_plus_frame_bbox, den_plus_frame_bbox)
                            # wasserstein_den_plus = wasserstein_dist(den_plus_frame_bbox, other_den_plus_frame_bbox)

                            kl_median = kl_divergence(other_median_frame_bbox, median_frame_bbox)
                            kl_gaussian = kl_divergence(other_gaussian_frame_bbox, gaussian_frame_bbox)
                            kl_n2v = kl_divergence(other_n2v_frame_bbox, n2v_frame_bbox)
                            kl_udvd = kl_divergence(other_udvd_frame_bbox, udvd_frame_bbox)

                            # raw_hellingers_clipwise[clip].append(hellinger_raw)
                            # bs_hellingers_clipwise[clip].append(hellinger_bs)
                            # bs_3_ch_hellingers_clipwise[clip].append(hellinger_bs_3_ch)
                            # den_hellingers_clipwise[clip].append(hellinger_den)
                            # den_plus_hellingers_clipwise[clip].append(hellinger_den_plus)

                            # raw_chi2_clipwise[clip].append(chi2_raw)
                            # bs_chi2_clipwise[clip].append(chi2_bs)
                            # bs_3_ch_chi2_clipwise[clip].append(chi2_bs_3_ch)
                            # den_chi2_clipwise[clip].append(chi2_den)
                            # den_plus_chi2_clipwise[clip].append(chi2_den_plus)

                            # raw_iou_clipwise[clip].append(iou_raw)
                            # bs_iou_clipwise[clip].append(iou_bs)
                            # bs_3_ch_iou_clipwise[clip].append(iou_bs_3_ch)
                            # den_iou_clipwise[clip].append(iou_den)
                            # den_plus_iou_clipwise[clip].append(iou_den_plus)

                            raw_kl_clipwise[clip].append(kl_raw)
                            bs_kl_clipwise[clip].append(kl_bs)
                            bs_3_ch_kl_clipwise[clip].append(kl_bs_3_ch)
                            den_kl_clipwise[clip].append(kl_den)
                            den_plus_kl_clipwise[clip].append(kl_den_plus)

                            # raw_wasserstein_clipwise[clip].append(wasserstein_raw)
                            # bs_wasserstein_clipwise[clip].append(wasserstein_bs)
                            # bs_3_ch_wasserstein_clipwise[clip].append(wasserstein_bs_3_ch)
                            # den_wasserstein_clipwise[clip].append(wasserstein_den)
                            # den_plus_wasserstein_clipwise[clip].append(wasserstein_den_plus)

                            median_kl_clipwise[clip].append(kl_median)
                            gaussian_kl_clipwise[clip].append(kl_gaussian)
                            n2v_kl_clipwise[clip].append(kl_n2v)
                            udvd_kl_clipwise[clip].append(kl_udvd)

                            count_total_frames += 1
                            # raw_total_hellingers += hellinger_raw
                            # raw_total_iou += iou_raw
                            raw_total_kl += kl_raw
                            # raw_total_chi2 += chi2_raw
                            # raw_total_wasserstein += wasserstein_raw
                            # bs_total_hellingers += hellinger_bs
                            # bs_total_iou += iou_bs
                            bs_total_kl += kl_bs
                            # bs_total_chi2 += chi2_bs
                            # bs_total_wasserstein += wasserstein_bs
                            # bs_3_ch_total_hellingers += hellinger_bs_3_ch
                            # bs_3_ch_total_iou += iou_bs_3_ch
                            bs_3_ch_total_kl += kl_bs_3_ch
                            # bs_3_ch_total_chi2 += chi2_bs_3_ch
                            # bs_3_ch_total_wasserstein += wasserstein_bs_3_ch
                            # den_total_hellingers += hellinger_den
                            # den_total_iou += iou_den
                            den_total_kl += kl_den
                            # den_total_chi2 += chi2_den
                            # den_total_wasserstein += wasserstein_den
                            # # print("Adding wasserstein_den", wasserstein_den)
                            # # print("Adding chi2_den", chi2_den)
                            # # print('='*20)
                            # den_plus_total_hellingers += hellinger_den_plus
                            # den_plus_total_iou += iou_den_plus
                            den_plus_total_kl += kl_den_plus
                            # den_plus_total_chi2 += chi2_den_plus
                            # den_plus_total_wasserstein += wasserstein_den_plus
                            median_total_kl += kl_median
                            gaussian_total_kl += kl_gaussian
                            n2v_total_kl += kl_n2v
                            udvd_total_kl += kl_udvd
                            # print("Adding wasserstein_den_plus", wasserstein_den_plus)
                            # print("Adding chi2_den_plus", chi2_den_plus)
                            continue
                    if not empty_box_found:
                        # print(f"No empty box found for frame {frame_idx} at location {bbox}")
                        pass
        # calculate the mean of the metrics for each clip to only store one value per clip
        # raw_hellingers_clipwise[clip] = np.mean(raw_hellingers_clipwise[clip])
        # bs_hellingers_clipwise[clip] = np.mean(bs_hellingers_clipwise[clip])
        # bs_3_ch_hellingers_clipwise[clip] = np.mean(bs_3_ch_hellingers_clipwise[clip])
        # den_hellingers_clipwise[clip] = np.mean(den_hellingers_clipwise[clip])
        # den_plus_hellingers_clipwise[clip] = np.mean(den_plus_hellingers_clipwise[clip])

        median_kl_clipwise[clip] = np.mean(median_kl_clipwise[clip])
        gaussian_kl_clipwise[clip] = np.mean(gaussian_kl_clipwise[clip])
        n2v_kl_clipwise[clip] = np.mean(n2v_kl_clipwise[clip])
        udvd_kl_clipwise[clip] = np.mean(udvd_kl_clipwise[clip])

        # raw_iou_clipwise[clip] = np.mean(raw_iou_clipwise[clip])
        # bs_iou_clipwise[clip] = np.mean(bs_iou_clipwise[clip])
        # bs_3_ch_iou_clipwise[clip] = np.mean(bs_3_ch_iou_clipwise[clip])
        # den_iou_clipwise[clip] = np.mean(den_iou_clipwise[clip])
        # den_plus_iou_clipwise[clip] = np.mean(den_plus_iou_clipwise[clip])

        raw_kl_clipwise[clip] = np.mean(raw_kl_clipwise[clip])
        bs_kl_clipwise[clip] = np.mean(bs_kl_clipwise[clip])
        bs_3_ch_kl_clipwise[clip] = np.mean(bs_3_ch_kl_clipwise[clip])
        den_kl_clipwise[clip] = np.mean(den_kl_clipwise[clip])
        den_plus_kl_clipwise[clip] = np.mean(den_plus_kl_clipwise[clip])

        # raw_chi2_clipwise[clip] = np.mean(raw_chi2_clipwise[clip])
        # bs_chi2_clipwise[clip] = np.mean(bs_chi2_clipwise[clip])
        # bs_3_ch_chi2_clipwise[clip] = np.mean(bs_3_ch_chi2_clipwise[clip])
        # den_chi2_clipwise[clip] = np.mean(den_chi2_clipwise[clip])
        # den_plus_chi2_clipwise[clip] = np.mean(den_plus_chi2_clipwise[clip])

        # raw_wasserstein_clipwise[clip] = np.mean(raw_wasserstein_clipwise[clip])
        # bs_wasserstein_clipwise[clip] = np.mean(bs_wasserstein_clipwise[clip])
        # bs_3_ch_wasserstein_clipwise[clip] = np.mean(bs_3_ch_wasserstein_clipwise[clip])
        # den_wasserstein_clipwise[clip] = np.mean(den_wasserstein_clipwise[clip])
        # den_plus_wasserstein_clipwise[clip] = np.mean(den_plus_wasserstein_clipwise[clip])

    # after looking at all the frames in all the clips, calculate the mean of the metrics
    # raw_total_hellingers = raw_total_hellingers / count_total_frames
    # raw_total_iou = raw_total_iou / count_total_frames
    raw_total_kl = raw_total_kl / count_total_frames
    # raw_total_chi2 = raw_total_chi2 / count_total_frames
    # raw_total_wasserstein = raw_total_wasserstein / count_total_frames
    # bs_total_hellingers = bs_total_hellingers / count_total_frames
    # bs_total_iou = bs_total_iou / count_total_frames
    bs_total_kl = bs_total_kl / count_total_frames
    # bs_total_chi2 = bs_total_chi2 / count_total_frames
    # bs_total_wasserstein = bs_total_wasserstein / count_total_frames
    # bs_3_ch_total_hellingers = bs_3_ch_total_hellingers / count_total_frames
    # bs_3_ch_total_iou = bs_3_ch_total_iou / count_total_frames
    bs_3_ch_total_kl = bs_3_ch_total_kl / count_total_frames
    # bs_3_ch_total_chi2 = bs_3_ch_total_chi2 / count_total_frames
    # bs_3_ch_total_wasserstein = bs_3_ch_total_wasserstein / count_total_frames
    # den_total_hellingers = den_total_hellingers / count_total_frames
    # den_total_iou = den_total_iou / count_total_frames
    den_total_kl = den_total_kl / count_total_frames
    # den_total_chi2 = den_total_chi2 / count_total_frames
    # den_total_wasserstein = den_total_wasserstein / count_total_frames
    # den_plus_total_hellingers = den_plus_total_hellingers / count_total_frames
    # den_plus_total_iou = den_plus_total_iou / count_total_frames
    den_plus_total_kl = den_plus_total_kl / count_total_frames
    # total_den_plus_total_chi2 = den_plus_total_chi2 / count_total_frames
    # total_den_plus_total_wasserstein = den_plus_total_wasserstein / count_total_frames
    median_total_kl = median_total_kl / count_total_frames
    gaussian_total_kl = gaussian_total_kl / count_total_frames
    n2v_total_kl = n2v_total_kl / count_total_frames
    udvd_total_kl = udvd_total_kl / count_total_frames

    print(f"Method & Hellinger & IoU & KL & Chi2 & Wasserstein")
    format_num = lambda x: f"{' '*8}{x:.4g} "
    
    # print(f"raw &{format_num(raw_total_hellingers)}&{format_num(raw_total_iou)}&{format_num(raw_total_kl)}&{format_num(raw_total_chi2)}&{format_num(raw_total_wasserstein)}")
    # print(f"bs &{format_num(bs_total_hellingers)}&{format_num(bs_total_iou)}&{format_num(bs_total_kl)}&{format_num(bs_total_chi2)}&{format_num(bs_total_wasserstein)}")
    # print(f"bs_3_ch &{format_num(bs_3_ch_total_hellingers)}&{format_num(bs_3_ch_total_iou)}&{format_num(bs_3_ch_total_kl)}&{format_num(bs_3_ch_total_chi2)}&{format_num(bs_3_ch_total_wasserstein)}")
    # print(f"den &{format_num(den_total_hellingers)}&{format_num(den_total_iou)}&{format_num(den_total_kl)}&{format_num(den_total_chi2)}&{format_num(den_total_wasserstein)}")
    # print(f"den_plus &{format_num(den_plus_total_hellingers)}&{format_num(den_plus_total_iou)}&{format_num(den_plus_total_kl)}&{format_num(total_den_plus_total_chi2)}&{format_num(total_den_plus_total_wasserstein)}")
    print(f"raw & {format_num(raw_total_kl)}")
    print(f"bs & {format_num(bs_total_kl)}")
    print(f"bs_3_ch & {format_num(bs_3_ch_total_kl)}")
    print(f"den & {format_num(den_total_kl)}")
    print(f"den_plus & {format_num(den_plus_total_kl)}")
    print(f"median &{format_num(median_total_kl)}")
    print(f"gaussian &{format_num(gaussian_total_kl)}")
    print(f"n2v &{format_num(n2v_total_kl)}")
    print(f"udvd &{format_num(udvd_total_kl)}")

    # print(f"clip with highest kl divergence: {clip_with_highest_kl_div} at frame {frame_with_highest_kl_div}: highest kl divergence diff: {highest_kl_div_diff}, highest kl divergence: {highest_kl_div}")

    # # create a dataframe to store the metrics
    # results = {
    #     'raw': {
    #         'hellinger': raw_hellingers_clipwise,
    #         'iou': raw_iou_clipwise,
    #         'kl': raw_kl_clipwise,
    #         'chi2': raw_chi2_clipwise,
    #         'wasserstein': raw_wasserstein_clipwise
    #     },
    #     'bs': {
    #         'hellinger': bs_hellingers_clipwise,
    #         'iou': bs_iou_clipwise,
    #         'kl': bs_kl_clipwise,
    #         'chi2': bs_chi2_clipwise,
    #         'wasserstein': bs_wasserstein_clipwise
    #     },
    #     'bs_3_ch': {
    #         'hellinger': bs_3_ch_hellingers_clipwise,
    #         'iou': bs_3_ch_iou_clipwise,
    #         'kl': bs_3_ch_kl_clipwise,
    #         'chi2': bs_3_ch_chi2_clipwise,
    #         'wasserstein': bs_3_ch_wasserstein_clipwise
    #     },
    #     'den': {
    #         'hellinger': den_hellingers_clipwise,
    #         'iou': den_iou_clipwise,
    #         'kl': den_kl_clipwise,
    #         'chi2': den_chi2_clipwise,
    #         'wasserstein': den_wasserstein_clipwise
    #     },
    #     'den_plus': {
    #         'hellinger': den_plus_hellingers_clipwise,
    #         'iou': den_plus_iou_clipwise,
    #         'kl': den_plus_kl_clipwise,
    #         'chi2': den_plus_chi2_clipwise,
    #         'wasserstein': den_plus_wasserstein_clipwise
    #     }
    # }
    # records = []
    # for method in results:
    #     for metric in results[method]:
    #         for clip in results[method][metric]:
    #             records.append({
    #                 'method': method,
    #                 'metric': metric,
    #                 'clip': clip,
    #                 'value': results[method][metric][clip]
    #             })
    # # Convert to DataFrame
    # df = pd.DataFrame(records)

    # # Save to CSV
    # df.to_csv('divergence_metrics.csv', index=False)
    # print('metrics saved to divergence_metrics.csv')




if __name__ == '__main__':
    args = parse_args()
    # main(args)
    # clips=['Elwha_2018_OM_ARIS_2018_07_16_2018-07-16_220000_2572_3023'] #  Elwha
    # clips = ['2018-05-31-JD151_LeftFar_Stratum2_Set1_LO_2018-05-31_101004_5_485'] # kenai-train
    # args.domain = 'kenai-train'
    
    # visualize_psnr_on_frames_fish_boxes_vs_frames_no_fish(args, clips)
    clips = os.listdir(os.path.join(args.den_plus_img_path, args.domain))
    # clips = ['2018-06-02-JD153_LeftNear_Stratum1_Set1_LN_2018-06-02_080000_0_422']
    # clips = ['Elwha_2018_OM_ARIS_2018_07_14_2018-07-14_020000_6293_6744']
    print(len(clips))
    # visualize_psnr_on_frames_fish_boxes_vs_frames_no_fish(args, clips)
    calculate_metrics_on_frames(args, clips)

