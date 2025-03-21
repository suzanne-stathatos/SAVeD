import yaml
import sys
import os
from pathlib import Path
from argparse import ArgumentParser
import shutil
from tqdm import tqdm
import pandas as pd
import cv2
import numpy as np

def add_noise_to_mask(mask, noise_level):
    # print(f"Adding noise to mask with level {noise_level}")
    # Add random amount of noise (in pixels) to the mask
    noise_amount = np.random.randint(0, int(noise_level * mask.size))
    # print(f"Noise amount: {noise_amount}")
    # Add noise_amount pixels to the mask
    for _ in range(noise_amount):
        x = np.random.randint(0, mask.shape[1])
        y = np.random.randint(0, mask.shape[0])
        mask[y, x] = np.random.randint(0, 256)
    return mask


def generate_noise_frame(img_height, img_width, noise_level):
    noise_frame = np.zeros((img_height, img_width), dtype=np.uint8)
    if noise_level > 0:
        noise_frame = add_noise_to_mask(noise_frame, noise_level)
    return noise_frame


# Generate using bounding box annos
parser = ArgumentParser()
parser.add_argument('--frames_dir', type=str, default='/path/to/Data/CFC22/frames/raw')
parser.add_argument('--anno_dir', type=str, default='/path/to/Data/CFC22/annotations_mot')
parser.add_argument('--save_dir', type=str, default='/path/to/Data/CFC22/frames/oracle_bbox_masks')
args = parser.parse_args()

original_data_location = Path(args.frames_dir)
original_annotations_location = Path(args.anno_dir)
save_dataset_location = Path(args.save_dir)

if save_dataset_location.exists():
    shutil.rmtree(save_dataset_location)
os.makedirs(save_dataset_location)

for domain in original_data_location.iterdir():
    if domain.is_dir() and not domain.name.startswith('.'):
        domain_save_location = save_dataset_location / domain.name
        os.makedirs(domain_save_location)

        domain_annotations_location = original_annotations_location / domain.name
        if not domain_annotations_location.exists():
            continue
        assert domain_annotations_location.exists(), f"Annotations location {domain_annotations_location} does not exist"

        for clip in tqdm(domain.iterdir(), desc=f'Processing domain: {domain.name}'):
            if clip.is_dir() and not clip.name.startswith('.'):
                clip_save_location = domain_save_location / clip.name
                os.makedirs(clip_save_location)

                anno_file = domain_annotations_location / clip.name / 'gt.txt'
                assert anno_file.exists(), f"Annotation file {anno_file} does not exist"

                # Load annotations file as a pandas dataframe
                anno_df = pd.read_csv(anno_file, header=None, names=['frame_idx', 'track_id', 'x1', 'y1', 'width', 'height', 'conf', 'x', 'y', 'z'])
                
                first_frame_seen = False
                # sort through frames in order
                frames = sorted(clip.glob('*.jpg'), key=lambda x: int(x.stem))
                print(f"Processing clip {domain.name}/{clip.name} with {len(frames)} frames")
                for frame_idx, frame in enumerate(frames):
                    if frame.is_file() and frame.name.endswith('.jpg'):
                        frame_save_location = clip_save_location / frame.name
                        # frame_im = cv2.imread(str(frame)) # DEBUG
                        if not first_frame_seen:
                            # load the first frame to get the image dimensions
                            first_frame = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE)
                            img_height, img_width = first_frame.shape
                            first_frame_seen = True
                        # create a black image of the same size
                        mask = np.zeros((img_height, img_width), dtype=np.uint8)
                        
                        # iterate through the annotations and draw the bounding boxes on the mask
                        annos_for_frame = anno_df[anno_df['frame_idx'] == frame_idx + 1 ] # +1 because MOT uses 1-indexed frame IDs
                        for _, anno in annos_for_frame.iterrows():
                            x1, y1, width, height = anno[['x1', 'y1', 'width', 'height']]
                            cv2.rectangle(mask, (int(x1), int(y1)), (int(x1 + width), int(y1 + height)), 255, -1) # fill in the bounding box with white

                        cv2.imwrite(str(frame_save_location), mask)


# Verify that the dataset produced the anticipated number of images
def verify_dataset_integrity(args):
    mask_location = Path(args.save_dir)

    assert mask_location.exists(), f"Mask location {mask_location} does not exist"

    # Check domain number matches
    frames_location = Path(args.frames_dir)
    original_dirs = [d for d in frames_location.iterdir() if d.is_dir() and not d.name.startswith('.')]
    mask_dirs = [d for d in mask_location.iterdir() if d.is_dir()]
    assert len(original_dirs) == len(mask_dirs), f"Number of directories mismatch: {len(mask_dirs)} in mask location, {len(original_dirs)} in original location"
    print(f"Verified: Both locations have the same number of domains.")


    # Check that the number of clips match
    for domain in original_dirs:
        original_clips = list(domain.glob('*'))
        mask_clips = list((mask_location / domain.name).glob('*'))
        assert len(original_clips) == len(mask_clips), f"Clip count mismatch in {domain.name}: {len(original_clips)} in original, {len(mask_clips)} in mask location"        
    print(f"Verified: Both locations have the same number of clips in each domain.")

    # Check that the number of files match within each clip
    for domain in original_dirs:
        original_clips = list(domain.glob('*'))
        mask_clips = list((mask_location / domain.name).glob('*'))

        for clip in original_clips:
            original_files = list(clip.glob('*'))
            mask_files = list((mask_location / domain.name / clip.name).glob('*'))
            assert len(original_files) == len(mask_files), f"File count mismatch in {clip.name}: {len(original_files)} in original, {len(mask_files)} in mask location"
    print("Verified: Both locations have the same number of files in each domain.")

verify_dataset_integrity(args)
