import os
import json
import cv2
from pathlib import Path
from collections import defaultdict

def get_image_dimensions(image_path):
    """Get dimensions of an image"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return img.shape[1], img.shape[0]  # width, height

def convert_mot_to_coco(mot_dir, frames_dir, output_json_path, pred=False):
    """
    Convert MOT format annotations to COCO format
    Args:
        mot_dir: Directory containing clip folders withMOT annotation files (gt.txt)
        frames_dir: Directory containing frame images
        output_json_path: Path to save COCO format JSON
    """
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "object"}]
    }
    if pred:
        mot_file_suffix = 'preds.txt'
    else:
        mot_file_suffix = 'gt.txt'
    
    annotation_id = 1
    image_id = 1
    
    # Process each clip
    clips = [d for d in os.listdir(mot_dir) if os.path.isdir(os.path.join(mot_dir, d))]
    print(f"Found {len(clips)} clips")
    
    for clip in clips:
        print(f"Processing clip: {clip}")
        mot_file = os.path.join(mot_dir, clip, mot_file_suffix)
        clip_frames_dir = os.path.join(frames_dir, clip)
        
        if not os.path.exists(mot_file):
            print(f"Warning: {mot_file} does not exist")
            continue
            
        # Group annotations by frame
        frame_annotations = defaultdict(list)
        with open(mot_file, 'r') as f:
            for line in f:
                frame, track_id, bb_left, bb_top, bb_width, bb_height, conf, _, _, _ = map(float, line.strip().split(','))
                frame = int(frame) - 1 # make it 0-indexed
                frame_annotations[frame].append({
                    'track_id': int(track_id) - 1, # make it 0-indexed
                    'bbox': [bb_left, bb_top, bb_width, bb_height],
                    'conf': conf
                })
        
        # Process each frame
        for frame in frame_annotations.keys():
            frame_path = os.path.join(clip_frames_dir, f'{frame}.jpg')
            if not os.path.exists(frame_path):
                print(f"Warning: Frame {frame_path} not found")
                continue
            
            # Get image dimensions
            try:
                width, height = get_image_dimensions(frame_path)
            except ValueError as e:
                print(f"Error with image {frame_path}: {e}")
                continue
            
            # Add image info
            coco_format["images"].append({
                "id": image_id,
                "file_name": f"{clip}/{frame}.jpg",
                "height": height,
                "width": width
            })
            
            # Add annotations
            for anno in frame_annotations[frame]:
                x, y, w, h = anno['bbox']
                
                # Ensure coordinates are within image bounds
                x = max(0, min(x, width))
                y = max(0, min(y, height))
                w = max(0, min(w, width - x))
                h = max(0, min(h, height - y))
                
                if pred:
                    coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "track_id": anno['track_id'], 
                    "score": anno['conf']
                    })
                else:
                    coco_format["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                        "track_id": anno['track_id'], 
                    })
                annotation_id += 1
            
            image_id += 1
    
    # Save COCO format annotations
    print(f"Saving annotations to {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(coco_format, f, indent=2)
    
    print(f"Converted {len(coco_format['images'])} images and {len(coco_format['annotations'])} annotations")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mot-dir', type=str, default='/path/to/Data/CFC22/annotations_mot', help='MOT annotations directory')
    parser.add_argument('--frames-dir', type=str, default='/path/to/Data/CFC22/frames/background_subtracted_frame_to_frame_difference', help='Frames directory')
    parser.add_argument('--output', type=str, default='/path/to/Data/CFC22/coco_dataset', help='Output folder')
    args = parser.parse_args()

    domains = os.listdir(args.mot_dir)
    os.makedirs(args.output, exist_ok=True)
    for domain in domains:
        mot_dir = os.path.join(args.mot_dir, domain)
        frames_dir = os.path.join(args.frames_dir, domain)
        output = os.path.join(args.output, f'{domain}.json')
        convert_mot_to_coco(mot_dir, frames_dir, output)
