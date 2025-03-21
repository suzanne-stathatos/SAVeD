import os
import cv2
import numpy as np
import shutil
from collections import defaultdict

MOT_ANNOTATIONS_PATH = '/path/to/Data/CFC22/annotations_mot'
FRAMES_PATH = '/path/to/Data/CFC22/frames/raw/'
YOLO_DATASET_PATH = '/path/to/Data/CFC22/yolo_dataset_raw'

train_and_val = {'train': 'kenai-train', 'val': 'kenai-val'}


def get_image_dimensions(image_path):
    """Get dimensions of an image"""
    img = cv2.imread(image_path)
    return img.shape[1], img.shape[0]  # width, height


def convert_mot_to_yolo(mot_file, frames_dir, output_images_dir, output_labels_dir, clip):
    """
    Convert MOT format annotations to YOLO format and organize files.
    MOT format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    YOLO format: <class> <x_center> <y_center> <width> <height>
    MOT format is 1-indexed, YOLO format is 0-indexed.
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # Read MOT file and group by frame
    # if clip != '2018-06-03-JD154_LeftNear_Stratum1_Set1_LN_2018-06-03_230000_4637_5018':
    #     return
    
    frame_annotations = defaultdict(list)
    with open(mot_file, 'r') as f:
        for line in f:
            frame, track_id, bb_left, bb_top, bb_width, bb_height, conf, _, _, _ = map(float, line.strip().split(','))
            frame = int(frame)
            
            # Store original coordinates for now
            # if frame == 253:
            #     print(f'{frame} {track_id} {bb_left} {bb_top} {bb_width} {bb_height}')
            frame_annotations[frame-1].append((bb_left, bb_top, bb_width, bb_height))
            # if frame == 253:
            #     print(frame_annotations[frame-1])
    
    # Process each frame
    for frame_0_indexed in frame_annotations.keys():
        # if frame_0_indexed != 252:
        #     continue
        # Get source image path
        src_img_path = os.path.join(frames_dir, f'{frame_0_indexed}.jpg')
        if not os.path.exists(src_img_path):
            print(f"Warning: Image {src_img_path} not found")
            continue
        
        # Get image dimensions
        img_width, img_height = get_image_dimensions(src_img_path)
        
        # Convert annotations to YOLO format
        yolo_annotations = []
        for bb_left, bb_top, bb_width, bb_height in frame_annotations[frame_0_indexed]:
            # Convert to YOLO format (normalized coordinates)
            x_center = (bb_left + bb_width/2.0) / img_width
            y_center = (bb_top + bb_height/2.0) / img_height
            width = bb_width / img_width
            height = bb_height / img_height
            
            yolo_annotations.append((0, x_center, y_center, width, height))  # 0 is the class ID

        # Symlink image to new location
        dst_img_path = os.path.join(output_images_dir, f'{clip}_{frame_0_indexed}.jpg')
        try:
            os.symlink(os.path.abspath(src_img_path), dst_img_path)
        except FileExistsError:
            # If link already exists, remove it and create new one
            os.remove(dst_img_path)
            os.symlink(os.path.abspath(src_img_path), dst_img_path)
        
        # Write YOLO annotations
        label_path = os.path.join(output_labels_dir, f'{clip}_{frame_0_indexed}.txt')
        with open(label_path, 'w') as f:
            for class_id, x_center, y_center, width, height in yolo_annotations:
                f.write(f'{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n')



def convert_dataset(mot_annotations_path, frames_base_path, output_base_path):
    """
    Convert entire dataset from MOT to YOLO format and organize into train/val splits
    """
    # Create output directories
    train_images_dir = os.path.join(output_base_path, 'images', 'train')
    train_labels_dir = os.path.join(output_base_path, 'labels', 'train')
    val_images_dir = os.path.join(output_base_path, 'images', 'val')
    val_labels_dir = os.path.join(output_base_path, 'labels', 'val')

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # Get clips in splits
    train_clips = os.listdir(os.path.join(mot_annotations_path, train_and_val['train']))
    val_clips = os.listdir(os.path.join(mot_annotations_path, train_and_val['val']))

    train_frames_base_path = os.path.join(frames_base_path, train_and_val['train'])
    val_frames_base_path = os.path.join(frames_base_path, train_and_val['val'])

    # Process train clips
    print("Processing training clips...")
    for clip in train_clips:
        print(f"Converting {clip}")
        mot_file = os.path.join(mot_annotations_path, train_and_val['train'], clip, 'gt.txt')
        frames_dir = os.path.join(train_frames_base_path, clip)
        
        if not os.path.exists(mot_file) or not os.path.exists(frames_dir):
            print(f"Warning: Missing files for {clip}, mot_file: {mot_file}, frames_dir: {frames_dir}")
            continue
            
        convert_mot_to_yolo(mot_file, frames_dir, train_images_dir, train_labels_dir, clip)
    
    # Process val clips
    print("\nProcessing validation clips...")
    for clip in val_clips:
        print(f"Converting {clip}")
        mot_file = os.path.join(mot_annotations_path, train_and_val['val'], clip, 'gt.txt')
        frames_dir = os.path.join(val_frames_base_path, clip)
        
        if not os.path.exists(mot_file) or not os.path.exists(frames_dir):
            print(f"Warning: Missing files for {clip}, mot_file: {mot_file}, frames_dir: {frames_dir}")
            continue
            
        convert_mot_to_yolo(mot_file, frames_dir, val_images_dir, val_labels_dir, clip)

    # Create data.yaml
    yaml_path = os.path.join(output_base_path, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {output_base_path}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("nc: 1\n")  # number of classes
        f.write("names: ['object']\n")  # class names


    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mot-dir', type=str, default=MOT_ANNOTATIONS_PATH, help='MOT annotations directory')
    parser.add_argument('--frames-dir', type=str, default=FRAMES_PATH, help='Directory containing frame images')
    parser.add_argument('--output-dir', type=str, default=YOLO_DATASET_PATH, help='Output directory for YOLO dataset')
    args = parser.parse_args()

    convert_dataset(args.mot_dir, args.frames_dir, args.output_dir)
