"""
Create a dataset with fraction of all frames from all domains, where the fraction is the same for each domain 
such that each domain has the same proportional number of frames as in the original dataset.

Alternatively, create_dataset_fractions_uniform.py creates a dataset with equal numbers of frames per domain if possible.
"""

import os
import argparse
import random
import torch
import numpy
from tqdm import tqdm
from collections import defaultdict

def create_fractional_dataset_by_domain(frames_dir, frames_output_dir, annotations_dir, annotations_output_dir, fraction, num_prev_frames=2):
    """
    Create symlinks for a fraction of frames from each domain while maintaining the same 
    proportional number of frames per domain as in the original dataset.
    """
    # Create output directories
    os.makedirs(frames_output_dir, exist_ok=True)
    os.makedirs(annotations_output_dir, exist_ok=True)
    
    # Get all domains
    domains = [d for d in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, d))]
    
    for domain in tqdm(domains, desc="Domains"):
        domain_path = os.path.join(frames_dir, domain)
        
        # Get all videos in domain
        videos = [v for v in os.listdir(domain_path) if os.path.isdir(os.path.join(domain_path, v))]

        # Create dictionary of video to frames
        frame_to_video = {}
        
        for video in tqdm(videos, desc="Creating video to frames dictionary", leave=False):
            video_path = os.path.join(domain_path, video)
            
            # Get all frames in video
            frames = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])
            full_frames = [os.path.join(video_path, f) for f in frames]
            for frame in full_frames:
                frame_to_video[frame] = video
        
        # Get all frames from domain
        all_frames = sorted(list(frame_to_video.keys()), 
                        key=lambda x: (frame_to_video[x], int(os.path.basename(x).split('.')[0])))
        total_frames = len(all_frames)
        target_total = int(total_frames * fraction)
        streamed_total = 0

        # Group frames by video
        video_frames = defaultdict(list)
        for frame in all_frames:
            video_frames[frame_to_video[frame]].append(frame)

        # Select continuous chunks from each video
        selected_frames = []
        for i, (video, frames) in enumerate(video_frames.items()):
            # Check if i is the last video
            if i == len(video_frames) - 1:
                num_frames = len(frames)
                num_frames_to_keep = target_total - streamed_total
            else:
                num_frames = len(frames)
                num_frames_to_keep = int(num_frames * fraction)
            
            # Ensure we keep enough frames for the denoising window
            if num_prev_frames == 2:
                min_chunk_size = 8 * num_prev_frames + 1  # frames needed for denoising
            elif num_prev_frames == 5:
                min_chunk_size = 4 * num_prev_frames + 1  # frames needed for denoising
            
            if num_frames_to_keep < min_chunk_size:
                print(f"Warning: Video {video} has too few frames to keep ({num_frames_to_keep}) " 
                  f"compared to minimum required ({min_chunk_size})")
                # Keep minimum required frames instead
                num_frames_to_keep = min_chunk_size
                
            # Calculate number of chunks and chunk size
            chunk_size = max(min_chunk_size, num_frames_to_keep)
            possible_start_indices = range(0, len(frames) - chunk_size + 1)
            
            # Randomly select start index
            if possible_start_indices:
                start_idx = random.choice(possible_start_indices)
                selected_frames.extend(frames[start_idx:start_idx + chunk_size])
            streamed_total = len(selected_frames)

        total_num_frames_selected = len(selected_frames)
        if fraction >= 0.1:
            assert total_num_frames_selected == target_total, f"Total number of frames selected ({total_num_frames_selected}) does not match target total ({target_total})"

        # Create symlinks for selected frames
        for frame in tqdm(selected_frames, desc="Symlinking Frames", leave=False):
            video = frame_to_video[frame]            
            # Create symlinks
            output_video_path = os.path.join(frames_output_dir)
            os.makedirs(output_video_path, exist_ok=True)
            src = os.path.abspath(frame)
            os.makedirs ( os.path.join(output_video_path, f"{domain}_{video}"), exist_ok=True)
            dst = os.path.join(output_video_path, f"{domain}_{video}/{os.path.basename(frame)}")
            if not os.path.exists(dst):
                os.symlink(src, dst)
        print(f"Created {total_num_frames_selected}/{total_frames} symlinks for {domain}")

        # Handle annotations
        # Create a mapping of frame numbers to new filenames
        frame_to_new_name = {}
        for frame in selected_frames:
            video = frame_to_video[frame]
            frame_num = int(os.path.basename(frame).split('.')[0])
            new_name = f"{domain}_{video}/{os.path.basename(frame)}"
            frame_to_new_name[(video, frame_num)] = new_name.split('.')[0]  # Store without extension
        
        # Process annotations for each video
        for video in tqdm(videos, desc="Processing Annotations", leave=False):
            anno_path = os.path.join(annotations_dir, video, 'gt.txt')
            if os.path.exists(anno_path):
                # Read annotations
                try:
                    with open(anno_path, 'r') as f:
                        lines = f.readlines()
                    
                    # Filter and modify annotations for selected frames
                    new_annotations = []
                    for line in lines:
                        parts = line.strip().split(',')
                        frame_num = int(parts[0])-1 # -1 because frame numbers start at 1 in the annotations but at 0 in the frames
                        
                        # Check if this frame was selected
                        video_path = os.path.join(domain_path, video)
                        frame_path = os.path.join(video_path, f"{frame_num}.jpg")
                        if frame_path in selected_frames:
                            # Update frame number to new filename
                            new_frame_name = frame_to_new_name[(video, frame_num)]
                            parts[0] = new_frame_name
                            new_annotations.append(','.join(map(str, parts)))
                    
                    # Save filtered annotations if there are any
                    if new_annotations:
                        output_anno_path = os.path.join(annotations_output_dir, 'gt.txt')
                        # Append to existing file or create new one
                        if os.path.exists(output_anno_path):
                            os.remove(output_anno_path)
                        with open(output_anno_path, 'w') as f:
                            f.write('\n'.join(new_annotations) + '\n')
                
                except Exception as e:
                    print(f"Error processing annotations for {video}: {e}")



parser = argparse.ArgumentParser(description="Create a dataset with all frames from all domains, using symlinks")
parser.add_argument("--frames_dir", type=str, default="/path/to/Data/CFC22/frames/raw", help="Frames directory")
parser.add_argument("--frames_output_dir", type=str, default="/path/to/Data/CFC22/frames/raw_all", help="Frames output directory")
parser.add_argument("--annotations_dir", type=str, default="/path/to/Data/CFC22/annotations_mot", help="Annotations directory")
parser.add_argument("--annotations_output_dir", type=str, default="/path/to/Data/CFC22/annotations_mot_all", help="Annotations output directory")
parser.add_argument("--fraction", type=float, default=0.5, 
                      choices=[0.02, 0.1, 0.25, 0.5],
                      help="Fraction of frames to keep (0.02, 0.1, 0.25, or 0.5)")
args = parser.parse_args()

# Append fraction to output directory name
if args.fraction == 0.5:
    suffix = "_half"
elif args.fraction == 0.25:
    suffix = "_quarter"
elif args.fraction == 0.1:
    suffix = "_tenth"
elif args.fraction == 0.02:
    suffix = "_hundredth"

FRAMES_OUTPUT_DIR = args.frames_output_dir + suffix
FRAMES_DIR = args.frames_dir
ANNOS_DIR = args.annotations_dir
ANNOS_OUTPUT_DIR = args.annotations_output_dir + suffix


# Set random seed (random, numpy, and torch seed are set to 0)
seed = 0
random.seed(seed)
torch.manual_seed(seed)
numpy.random.seed(seed)

create_fractional_dataset_by_domain(FRAMES_DIR, 
                                    FRAMES_OUTPUT_DIR, 
                                    ANNOS_DIR, 
                                    ANNOS_OUTPUT_DIR, 
                                    args.fraction)

# Verify
# check that the total number of frames in all subdirectories of FRAMES_OUTPUT_DIR is the same as the sum of the frames in each domain
total_frames_linked = 0
total_frames_og = 0

# Count linked frames
for video in os.listdir(FRAMES_OUTPUT_DIR):
    video_path = os.path.join(FRAMES_OUTPUT_DIR, video)
    total_frames_linked += len(os.listdir(video_path))

# Count original frames
for domain in os.listdir(FRAMES_DIR):
    domain_path = os.path.join(FRAMES_DIR, domain)
    if not os.path.isdir(domain_path):
        continue
        
    for video in os.listdir(domain_path):
        video_path = os.path.join(FRAMES_DIR, domain, video)
        try:
            if os.path.isdir(video_path):
                total_frames_og += len(os.listdir(video_path))
        except OSError as e:
            print(f'Error counting frames in {video}: {e}')

print(f'Total number of frames in symlink: {total_frames_linked}')
print(f'Total number of frames in original: {total_frames_og}')
print(f'Fraction: {args.fraction}')
if args.fraction >= 0.1:
    assert total_frames_linked in range(int(total_frames_og * args.fraction) - 2, int(total_frames_og * args.fraction) + 2), "Total number of frames in symlink and original do not match"

