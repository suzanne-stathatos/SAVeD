import os
from argparse import ArgumentParser
import cv2
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm

from Tracker import Tracker
from tracker_bytetrack import Associate
# sys.path.append('/root/sort')
# from sort import Sort

### Configuration options
# from https://huggingface.co/spaces/caltech-fish-counting/fisheye-advanced/blob/main/backend/inference.py
# and 
# https://huggingface.co/spaces/caltech-fish-counting/fisheye-advanced/blob/main/backend/InferenceConfig.py
BATCH_SIZE = 32
CONF_THRES = 0.05 # detection
NMS_IOU  = 0.25 # NMS IOU
MAX_AGE = 20 # time until missing fish get's new id
MIN_HITS = 11 # minimum number of frames with a specific fish for it to count
IOU_THRES = 0.01 # IOU threshold for tracking
MIN_TRAVEL = 0 # Minimum distance a track has to travel
CONF_THRES_LOW = 0.1  # low confidence threshold for ByteTrack
CONF_THRES_HIGH = 0.3  # high confidence threshold for ByteTrack


# def run_sort ( clip_detections_file, track_file, img_width, img_height):
#     print(f"Running sort on {clip_detections_file} to {track_file}")
        
#     # initialize Sort with default parameters to start
#     tracker = Sort(max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THRES)

#     # Initialize how we'll write output in MOTChallenge format
#     MOTChallengeOutput = []
    
#     # Iterate over all predictions starting with minimum frame number going to maximum frame number
#     if not os.path.exists(clip_detections_file):
#         detections = []
#     else:
#         # load predictions
#         with open ( clip_detections_file, 'r' ) as f:
#             # Organize predictions by frame number
#             all_nms_pred_strings = f.readlines()
#             all_nms_preds = []
#             for line in all_nms_pred_strings:
#                 pred = line.split(',')
#                 # convert all pred values to float
#                 pred = [float(p) for p in pred]
#                 all_nms_preds.append(pred)
#             # Make dictionary of frame number to list of predictions
#             frame_to_preds = {}
#             for pred in all_nms_preds:
#                 frame_num = int(pred[0])
#                 bbox = pred[2:7]
#                 if frame_num not in frame_to_preds:
#                     frame_to_preds[frame_num] = []
#                 frame_to_preds[frame_num].append(bbox)
                        
#             # # filter detections
#             # if CONF_THRES > 0:
#             #     all_nms_preds = [ pred for pred in all_nms_preds if float(pred[6]) > CONF_THRES]
#             min_frame = min(frame_to_preds.keys())
#             max_frame = max(frame_to_preds.keys())
#             for frame_num in range(min_frame, max_frame+1):
#                 if frame_num not in frame_to_preds:
#                     tracked_objects = tracker.update(np.empty((0,6)))
#                     continue
#                 else:
#                     detections = frame_to_preds[frame_num]
#                     det_array = np.array([[
#                                     d[0],          # x1
#                                     d[1],          # y1
#                                     d[0] + d[2],   # x2 = x1 + w
#                                     d[1] + d[3],   # y2 = y1 + h
#                                     d[4]           # confidence score
#                                 ] for d in detections])
#                     tracked_objects = tracker.update(det_array)                    
        
#                 # write in MOTChallenge format 
#                 # NOTE: x,y,z are all -1
#                 # NOTE: frame_number, track_ids, bboxes are all 1-indexed
#                 # <frame_number>, <track_id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
#                 frame_num_MOT = frame_num
#                 for track in tracked_objects:
#                     x1, y1, x2, y2, conf, track_id = track
#                     width = x2 - x1
#                     height = y2 - y1
#                     MOTChallengeOutput.append([frame_num_MOT, int(track_id)+1, int(x1)+1, int(y1)+1, int(width), int(height), conf, -1,-1,-1 ])

#     # write MOTChallengeOutput lines all to a single file
#     with open ( track_file, 'w' ) as f:
#         for entry in MOTChallengeOutput:
#             line = ','.join(map(str, entry))
#             line_with_newline = line + '\n'
#             f.write ( line_with_newline )
#     print ( f"Output written to {track_file}" )


def load_detections(detection_file):
    """Load detections from MOT format file"""
    if not os.path.exists(detection_file):
        print(f"No detections file found at {detection_file}")
        return {}

    # Read detections file
    dets_df = pd.read_csv(detection_file, header=None, 
                         names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'x3d', 'y3d', 'z3d'])
    
    # Convert to dictionary of frame -> detections
    detections = {}
    for frame_id, frame_dets in dets_df.groupby('frame'):
        # Convert each detection to [x1, y1, x2, y2, conf] format
        boxes = []
        for _, det in frame_dets.iterrows():
            if det.conf >= CONF_THRES_LOW:  # Filter by minimum confidence
                x1 = det.x
                y1 = det.y
                x2 = det.x + det.w
                y2 = det.y + det.h
                boxes.append([x1, y1, x2, y2, det.conf])
        
        if boxes:
            detections[frame_id] = np.array(boxes)
        else:
            detections[frame_id] = None
            
    return detections

def run_bytetrack(clip_detections_file, track_file, img_width, img_height):
    """Run ByteTrack on detections file"""
    print(f"Running ByteTrack on {clip_detections_file}")
    
    # Load all detections
    all_detections = load_detections(clip_detections_file)
    if not all_detections:
        return
    
    # Get frame range
    frame_nums = sorted(all_detections.keys())
    min_frame = min(frame_nums)
    max_frame = max(frame_nums)
    
    # Initialize tracker
    clip_info = {
        'start_frame': min_frame,
        'end_frame': max_frame,
        'image_meter_width': img_width,
        'image_meter_height': img_height
    }
    
    tracker = Tracker(clip_info, algorithm=Associate, 
                     args={'max_age': MAX_AGE, 'min_hits': MIN_HITS, 'iou_threshold': IOU_THRES})
    
    # Store confidences by track ID and frame
    confidence_map = {}  # frame_num -> {bbox -> confidence}
    
    # Process each frame
    with tqdm(total=len(frame_nums), desc="Tracking") as pbar:
        for frame_num in frame_nums:
            dets = all_detections[frame_num]
            if dets is None:
                dets = np.empty((0, 5))
            
            # Store confidences for this frame
            confidence_map[frame_num] = {}
            for det in dets:
                bbox_key = tuple(det[:4])  # Use bbox coordinates as key
                confidence_map[frame_num][bbox_key] = det[4]
            
            # Split into high/low confidence detections for ByteTrack
            high_mask = dets[:, 4] >= CONF_THRES_HIGH
            high_dets = dets[high_mask]
            low_dets = dets[~high_mask]
            
            # Update tracker
            tracker.update((low_dets, high_dets), frame_num=frame_num)
            pbar.update(1)
    
    # Get final tracks
    results = tracker.finalize()
    
    # Write results in MOT format
    mot_rows = []
    for frame in results['frames']:
        frame_num = frame['frame_num']
        for fish in frame['fish']:
            bbox = fish['bbox']
            
            # Try to find matching detection in this frame
            bbox_key = tuple(bbox)
            conf = -1
            
            # Look for closest bbox in this frame's detections
            if frame_num in confidence_map:
                min_iou_dist = float('inf')
                best_conf = -1
                for det_bbox, det_conf in confidence_map[frame_num].items():
                    # Calculate IoU distance
                    det_bbox = np.array(det_bbox)
                    curr_bbox = np.array(bbox)
                    iou = calculate_iou(det_bbox, curr_bbox)
                    if iou > 0.5:  # If IoU is good enough, use this confidence
                        best_conf = det_conf
                        break
                conf = best_conf
            
            row = [
                str(frame_num),  # frame number
                str(fish['fish_id'] + 1),  # track ID (1-based)
                str(int(bbox[0])),  # x
                str(int(bbox[1])),  # y
                str(int(bbox[2] - bbox[0])),  # width
                str(int(bbox[3] - bbox[1])),  # height
                str(conf),  # confidence
                "-1", "-1", "-1"  # 3D coordinates (not used)
            ]
            mot_rows.append(",".join(row))
    
    # Write output file
    if os.path.exists(track_file):
        os.remove(track_file)
    with open(track_file, 'w') as f:
        f.write("\n".join(mot_rows))
    
    print(f"Tracking results written to {track_file}")

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in [x1,y1,x2,y2] format"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset',  type=str, default='kenai-val', help='Dataset name')
    parser.add_argument('--detections_dir', type=str, help='Directory with detection results', required=True)
    parser.add_argument('--frames_dir', type=str, help='Directory with frames', required=True)
    parser.add_argument('--nms_thresh', type=float, default=NMS_IOU, help='NMS threshold for evaluation')
    parser.add_argument('--metadata', type=str, help='Directory with metadata', required=True)
    args = parser.parse_args()

    DEBUG=False

    # Validate paths 
    for path in [args.detections_dir]:
        assert os.path.exists(path), f"Path {path} does not exist"


    output_dir = os.path.join(args.detections_dir, args.dataset)
    output_file = 'tracks.txt'
    print(f"Detection predictions loading from {args.detections_dir}")
    clip_detections = os.listdir(output_dir)

    if DEBUG:
        clip_detections = ['2018-06-03-JD154_LeftFar_Stratum1_Set1_LO_2018-06-03_050004_319_919']

    for clip in clip_detections:
        if clip.endswith('.txt') or clip.endswith('.csv'):
            continue
        clip_detections_file = os.path.join(output_dir, clip, 'preds.txt')
        if not os.path.exists(clip_detections_file):
            print(f"Skipping {clip_detections_file} because it does not exist")
            continue
        track_file = os.path.join(output_dir, clip, output_file)
        # Get first frame to get the img size
        first_frame = os.path.join(args.frames_dir, args.dataset,clip, '10.jpg')
        assert os.path.exists(first_frame), f"File {first_frame} does not exist"
        img = cv2.imread(first_frame)
        img_size = img.shape
        print(f"Processing {clip_detections_file} to {track_file}")
        img_width, img_height = img_size[1], img_size[0]
        # run_sort(clip_detections_file, track_file, img_width, img_height)
        run_bytetrack(clip_detections_file, track_file, img_width, img_height)
        print(f"Processed {clip_detections_file} to {track_file}")
    
if __name__ == "__main__":
    main()