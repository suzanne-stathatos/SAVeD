import os
import re
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages
from argparse import ArgumentParser

import torchmetrics
import torch
import json

def get_all_frame_numbers(image_dir):
    """Get sorted list of all frame numbers from image files"""
    frames = []
    for f in os.listdir(image_dir):
        if f.endswith(('.jpg', '.png')):
            frame_num = int(f.split('.')[0])
            frames.append(frame_num)
    return sorted(frames)


def save_as_gif_pil(frames, output_path, fps=20):
    """
    frames: list of numpy arrays
    """
    pil_frames = [Image.fromarray(frame) for frame in frames]
    duration = 1000 / fps  # Convert fps to milliseconds per frame
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0
    )


class DetectorEvaluator:
    def __init__(self, args):
        self.args = args
        self.pred_dir = os.path.join(args.output_dir, args.dataset)
        self.gt_dir = os.path.join(args.gt_dir, args.dataset)
        self.input_dir = os.path.join(args.input_dir, args.dataset)
        self.output_dir = args.output_dir
        self.raw_dir = os.path.join(args.raw_dir, args.dataset)
        self.mp4_dir = os.path.join(args.output_dir, f'{args.dataset}_mp4s')
        os.makedirs(self.mp4_dir, exist_ok=True)
        self.dataset = args.dataset
        self.ap_thresh = args.ap_thresh
        self.nms_thresh = args.nms_thresh
        self.create_mp4s = args.create_mp4s
        self.create_gifs = args.create_gifs
        self.pdf = None
        self.predictions = {}  # Store all predictions
        self.matched_predictions = {}  # Store which predictions were TPs
        self.coco_gt = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'fish'}]}
        self.coco_dt = []
        self.image_id = 0
        self.ann_id = 0

        # Initialize torchmetrics map
        self.metric = torchmetrics.detection.MeanAveragePrecision(
            box_format='xywh',  # bbox format
            iou_thresholds=[0.5],  # Only use IoU threshold of 0.5
            rec_thresholds=list(np.linspace(0, 1, 101)),  # Convert to list of floats
            max_detection_thresholds=[1, 10, 100]  # Required 3 values: small, medium, large
        )

    def evaluate(self):
        print("Starting evaluation...")
        clips = self._get_clips()
        self._evaluate_clips(clips)
        results = self._calculate_metrics()
        print(f'mAP50: {results["mAP50"]:.3f}')
        print(f'Precision: {results["precision"]:.3f}')
        print(f'Recall: {results["recall"]:.3f}')


    def _get_clips(self):
        """Get list of clips to evaluate"""
        clips = [f for f in os.listdir(self.pred_dir) if os.path.isdir(os.path.join(self.pred_dir, f))]
        print(f"Found {len(clips)} clips")
        return clips
    

    def _get_available_frames(self, clip):
        """Get set of frames that have corresponding images"""
        frames_dir = os.path.join(self.input_dir, clip)
        available_frames = set(int(f.split('.')[0]) 
                             for f in os.listdir(frames_dir) 
                             if f.endswith('.jpg'))
        if not available_frames:
            raise ValueError(f"No jpg frames found in {frames_dir}")
        return available_frames


    def _load_annotations(self, file_path, available_frames):
        """Load and verify annotation format"""
        annotations = defaultdict(dict)
        
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 7:  # Ensure we have all needed fields
                    frame_idx = int(parts[0]) - 1 # Adjust for 0-based indexing
                    assert frame_idx >=0, f"Frame index {frame_idx} is negative"
                    if frame_idx in available_frames:
                        obj_id = int(parts[1])
                        # Create bbox with x,y,w,h,conf
                        bbox = [float(x) for x in parts[2:7]]
                        annotations[frame_idx][obj_id] = bbox
        return annotations


    def _calculate_metrics(self):
        """Calculate metrics using torchmetrics"""
        results = self.metric.compute()
        return {
            'mAP50': results['map_50'].item(),  # AP at IoU=.50
            'precision': results['map_50'].item(),  # Using mAP as precision
            'recall': results['mar_100'].item()  # Mean Average Recall
        }

    def _evaluate_clip(self, clip):
        """Convert clip data to torch format and evaluate"""
        gt_file = os.path.join(self.gt_dir, clip, 'gt.txt')
        pred_file = os.path.join(self.pred_dir, clip, 'preds.txt')
        available_frames = self._get_available_frames(clip)

        # Load annotations
        predictions = self._load_annotations(pred_file, available_frames)
        ground_truths = self._load_annotations(gt_file, available_frames)

        # Process each frame
        for frame_id in available_frames:
            # Format predictions for torchmetrics
            preds_list = []
            if frame_id in predictions:
                boxes = []
                scores = []
                labels = []  # All labels are 0 (single class)
                for obj_id, bbox in predictions[frame_id].items():
                    x, y, w, h, score = bbox
                    boxes.append([x, y, w, h])
                    scores.append(score)
                    labels.append(0)
                if boxes:  # Only add if there are predictions
                    preds_list = {
                        'boxes': torch.tensor(boxes),
                        'scores': torch.tensor(scores),
                        'labels': torch.tensor(labels)
                    }

            # Format ground truth for torchmetrics
            target_list = []
            if frame_id in ground_truths:
                boxes = []
                labels = []
                for obj_id, bbox in ground_truths[frame_id].items():
                    x, y, w, h = bbox[:4]
                    boxes.append([x, y, w, h])
                    labels.append(0)
                if boxes:  # Only add if there are ground truths
                    target_list = {
                        'boxes': torch.tensor(boxes),
                        'labels': torch.tensor(labels)
                    }

            # Update metric
            if preds_list and target_list:  # Only update if both exist
                self.metric.update([preds_list], [target_list])


    def _evaluate_clips(self, clips):
        """Evaluate all clips"""
        results = []
        for clip in clips:
            # print(f"\nProcessing clip: {clip}")
            self._evaluate_clip(clip)
        


def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset',  type=str, default='kenai-train', help='Dataset name')
    parser.add_argument('--det_model', type=str, default='yolo_train/denoised_output_896/weights/best.pt', help='Model path')
    parser.add_argument('--input_dir', type=str, help='Input directory', required=True)
    parser.add_argument('--raw_dir', type=str, help='Directory of raw frames', required=True)
    parser.add_argument('--output_dir', type=str, help='Output directory', required=True)
    parser.add_argument('--gt_dir', type=str, help='Ground truth directory', required=True)
    parser.add_argument('--ap_thresh', type=float, default=0.01, help='AP threshold for evaluation')
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='NMS threshold for evaluation')
    parser.add_argument('--create_mp4s', action='store_true', help='Create mp4s for worst clips')
    parser.add_argument('--create_gifs', action='store_true', help='Create gifs for all clips')
    args = parser.parse_args()
    
    output_dir = os.path.join(args.output_dir, args.dataset)
    gt_dir = os.path.join(args.gt_dir, args.dataset)
    input_dir = os.path.join(args.input_dir, args.dataset)
    print(f"Detection predictions loading from {output_dir}, frames from {input_dir}, ground truth from {gt_dir}")

    # Validate paths 
    for path in [output_dir, gt_dir, input_dir]:
        assert os.path.exists(path), f"Path {path} does not exist"

    # Run eval for the dataset
    evaluator = DetectorEvaluator(args)
    evaluator.evaluate()

if __name__ == "__main__":
    main()