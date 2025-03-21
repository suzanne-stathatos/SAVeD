"""
Create a dataset with all frames from all domains, using symlinks
"""

import os
import argparse

parser = argparse.ArgumentParser(description="Create a dataset with all frames from all domains, using symlinks")
parser.add_argument("--frames_dir", type=str, default="/path/to/Data/CFC22/frames/raw", help="Frames directory")
parser.add_argument("--frames_output_dir", type=str, default="/path/to/Data/CFC22/frames/raw_all", help="Frames output directory")
parser.add_argument("--annotations_dir", type=str, default="/path/to/Data/CFC22/annotations_mot", help="Annotations directory")
parser.add_argument("--annotations_output_dir", type=str, default="/path/to/Data/CFC22/annotations_mot_all", help="Annotations output directory")
args = parser.parse_args()

FRAMES_DIR = args.frames_dir
FRAMES_OUTPUT_DIR = args.frames_output_dir
ANNOS_DIR = args.annotations_dir
ANNOS_OUTPUT_DIR = args.annotations_output_dir

os.makedirs(FRAMES_OUTPUT_DIR, exist_ok=True)
if os.path.exists(ANNOS_DIR):
    os.makedirs(ANNOS_OUTPUT_DIR, exist_ok=True)

# iterate through all domains in FRAMES_DIR and create symlinks to all videos in FRAMES_OUTPUT_DIR
for domain in os.listdir(FRAMES_DIR):
    domain_path = os.path.join(FRAMES_DIR, domain)
    if not os.path.isdir(domain_path):
        continue
        
    for video in os.listdir(domain_path):
        original_video_path = os.path.join(FRAMES_DIR, domain, video)
        symlinked_video_path = os.path.join(FRAMES_OUTPUT_DIR, f'{domain}_{video}')
        
        if os.path.isdir(original_video_path):
            try:
                # Remove existing symlink if it exists but is broken
                if os.path.islink(symlinked_video_path) and not os.path.exists(symlinked_video_path):
                    os.remove(symlinked_video_path)
                
                # Create symlink if it doesn't exist
                if not os.path.exists(symlinked_video_path):
                    os.symlink(
                        os.path.abspath(original_video_path),  # Use absolute path
                        symlinked_video_path
                    )
                    print(f'Created symlink for {video}')
            except OSError as e:
                print(f'Error creating symlink for {video}: {e}')
        else:
            print(f'Skipping {video} because it is not a directory')

# iterate through all domains in ANNOS_DIR and create symlinks to all videos in ANNOS_OUTPUT_DIR
if os.path.exists(ANNOS_DIR):
    for domain in os.listdir(ANNOS_DIR):
        domain_path = os.path.join(ANNOS_DIR, domain)
        if not os.path.isdir(domain_path):
            continue
            
        for video in os.listdir(domain_path):
            original_video_path = os.path.join(ANNOS_DIR, domain, video)
            symlinked_video_path = os.path.join(ANNOS_OUTPUT_DIR, f'{domain}_{video}')
            
            if os.path.isdir(original_video_path):
                try:
                    # Remove existing symlink if it exists but is broken
                    if os.path.islink(symlinked_video_path) and not os.path.exists(symlinked_video_path):
                        os.remove(symlinked_video_path)
                    
                    # Create symlink if it doesn't exist
                    if not os.path.exists(symlinked_video_path):
                        os.symlink(
                            os.path.abspath(original_video_path),  # Use absolute path
                            symlinked_video_path
                        )
                        print(f'Created annotation symlink for {video}')
                except OSError as e:
                    print(f'Error creating annotation symlink for {video}: {e}')
            else:
                print(f'Skipping annotation {video} because it is not a directory')


# Verify
# check that the total number of frames in all subdirectories of FRAMES_OUTPUT_DIR is the same as the sum of the frames in each domain
total_frames_linked = 0
total_frames_og = 0

# Count linked frames
for video in os.listdir(FRAMES_OUTPUT_DIR):
    video_path = os.path.join(FRAMES_OUTPUT_DIR, video)
    try:
        if os.path.isdir(video_path):
            total_frames_linked += len(os.listdir(video_path))
    except OSError as e:
        print(f'Error counting frames in {video}: {e}')

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
assert total_frames_linked == total_frames_og, "Total number of frames in symlink and original do not match"

if os.path.exists(ANNOS_DIR):
    # Verify that the annotations are the same
    total_annotations_linked = 0
    total_annotations_og = 0

    # Count linked annotations
    for video in os.listdir(ANNOS_OUTPUT_DIR):
        video_path = os.path.join(ANNOS_OUTPUT_DIR, video)
        try:
            if os.path.isdir(video_path):
                total_annotations_linked += len(os.listdir(video_path))
        except OSError as e:
            print(f'Error counting annotations in {video}: {e}')

    # Count original annotations
    for domain in os.listdir(ANNOS_DIR):
        domain_path = os.path.join(ANNOS_DIR, domain)
        if not os.path.isdir(domain_path):
            continue
            
        for video in os.listdir(os.path.join(ANNOS_DIR, domain)):
            video_path = os.path.join(ANNOS_DIR, domain, video)
            try:
                if os.path.isdir(video_path):
                    total_annotations_og += len(os.listdir(video_path))
            except OSError as e:
                print(f'Error counting annotations in {video}: {e}')

    print(f'Total number of annotations in symlink: {total_annotations_linked}')
    print(f'Total number of annotations in original: {total_annotations_og}')
    assert total_annotations_linked == total_annotations_og, "Total number of annotations in symlink and original do not match"
