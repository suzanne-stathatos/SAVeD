import pandas as pd
import numpy as np
import os
__all__ = ['get_motion_from_mot', 'get_trajectory_mse', 'spatial_min_max_normalize', 'transform_bbox_annos']


def spatial_min_max_normalize(img):
    min_val = img.min()
    max_val = img.max()
    if max_val == min_val:
        # The image is constant, return array of zeros
        return np.zeros_like(img)
    return (img - min_val) / (max_val - min_val)



def transform_bbox_annos(mot_annotations, original_shape, transform_shape):
    ''' 
    MOT annotations are in the original shape of the image.
    This function transforms the annotations to the shape of the transformed image.
    '''
    if len(mot_annotations) == 0:
        return mot_annotations
        
    tr_c, tr_h, tr_w = transform_shape
    w, h = original_shape
    
    # Vectorized transformation
    scale_x = tr_w / w
    scale_y = tr_h / h
    
    mot_annotations = mot_annotations.copy()
    mot_annotations.loc[:, 'x1'] *= scale_x
    mot_annotations.loc[:, 'y1'] *= scale_y
    mot_annotations.loc[:, 'w'] *= scale_x
    mot_annotations.loc[:, 'h'] *= scale_y
    
    return mot_annotations


def calculate_velocities(fish_df):
    """Calculate velocities from positions"""
    # Make explicit copy to avoid warnings
    fish_df = fish_df.copy()
    
    # Calculate differences using .loc
    fish_df.loc[:, 'dx'] = fish_df['x1'].diff()
    fish_df.loc[:, 'dy'] = fish_df['y1'].diff()
    
    # Calculate velocity
    fish_df.loc[:, 'velocity'] = np.sqrt(fish_df['dx']**2 + fish_df['dy']**2)
    
    return fish_df

def get_motion_from_mot(mot_path):
    """
    From an MOT file for a clip, get a score for the amount of motion within that clip from the fish.
    For now, it will be the average of average velocity of each fish in the clip.
    """
    # Basic file checks
    if not os.path.exists(mot_path):
        raise FileNotFoundError(f"File not found: {mot_path}")
        
    if os.path.getsize(mot_path) == 0:
        print(f"WARNING Empty file: {mot_path}")
        return 0
    
    # Read the MOT file
    df = pd.read_csv(mot_path)
    df.columns = ['frame', 'id', 'x1', 'y1', 'w', 'h', 'score', 'x', 'y', 'z']

    velocities = []

    # Get each fish's velocity 
    unique_fish_ids = df['id'].unique()
    for fish_id in unique_fish_ids:
        fish_df = df[df['id'] == fish_id]
        fish_df = calculate_velocities(fish_df)
        v = np.mean(fish_df['velocity'])
        velocities.append(v)
    return np.mean(velocities)


def get_size_from_mot(mot_path):
    """
    From an MOT file for a clip, get a score for the average size of the fish in the clip.
    """
    if not os.path.exists(mot_path):
        raise FileNotFoundError(f"File not found: {mot_path}")
        
    if os.path.getsize(mot_path) == 0:
        print(f"WARNING Empty file: {mot_path}")
        return 0
    
    df = pd.read_csv(mot_path)
    df.columns = ['frame', 'id', 'x1', 'y1', 'w', 'h', 'score', 'x', 'y', 'z']
    max_width = np.max(df['w'])
    max_height = np.max(df['h'])
    return max_width * max_height


def get_avg_conf_from_mot(mot_path):
    """
    From an MOT file for a clip, get a score for the average confidence of the fish in the clip.
    """
    if not os.path.exists(mot_path):
        raise FileNotFoundError(f"File not found: {mot_path}")
        
    if os.path.getsize(mot_path) == 0:
        print(f"WARNING Empty file: {mot_path}")
        return 0
    
    df = pd.read_csv(mot_path)
    df.columns = ['frame', 'id', 'x1', 'y1', 'w', 'h', 'score', 'x', 'y', 'z']
    return np.mean(df['score'])


def get_trajectory_mse(gt_mot_file, pred_mot_file):
    """
    Get the mean squared error between the ground truth trajectory and the predicted trajectory.
    """
    if not os.path.exists(pred_mot_file):
        raise FileNotFoundError(f"File not found: {pred_mot_file}")
        
    if os.path.getsize(pred_mot_file) == 0:
        print(f"WARNING Empty file: {pred_mot_file}")
        return 0
    
    gt_df = pd.read_csv(gt_mot_file)
    pred_df = pd.read_csv(pred_mot_file)

    gt_df.columns = ['frame', 'id', 'x1', 'y1', 'w', 'h', 'score', 'x', 'y', 'z']
    pred_df.columns = ['frame', 'id', 'x1', 'y1', 'w', 'h', 'score', 'x', 'y', 'z']
    gt_df['centroid_x'] = gt_df['x1'] + gt_df['w'] / 2
    gt_df['centroid_y'] = gt_df['y1'] + gt_df['h'] / 2
    pred_df['centroid_x'] = pred_df['x1'] + pred_df['w'] / 2
    pred_df['centroid_y'] = pred_df['y1'] + pred_df['h'] / 2

    # Get the unique fish IDs in the ground truth and predicted data
    gt_fish_ids = gt_df['id'].unique()
    pred_fish_ids = pred_df['id'].unique()

    # Calculate the MSE for each fish from the GT
    mse = 0
    for fish_id in gt_fish_ids:
        gt_fish_df = gt_df[gt_df['id'] == fish_id]
        pred_fish_df = pred_df[pred_df['id'] == fish_id]

        # Calculate the MSE for each fish
        mse += np.mean((gt_fish_df['centroid_x'] - pred_fish_df['centroid_x'])**2 + (gt_fish_df['centroid_y'] - pred_fish_df['centroid_y'])**2)

    # NOTE: Does not penalize for extra fish in the predicted data

    return mse / len(gt_fish_ids)

def get_trajectory_l2_no_fp_penalty(gt_mot_file, pred_mot_file):
    """
    Get the mean squared error between the ground truth trajectory and the predicted trajectory.
    """
    if not os.path.exists(pred_mot_file):
        raise FileNotFoundError(f"File not found: {pred_mot_file}")
        
    if os.path.getsize(pred_mot_file) == 0:
        print(f"WARNING Empty file: {pred_mot_file}")
        return 0
    
    gt_df = pd.read_csv(gt_mot_file)
    pred_df = pd.read_csv(pred_mot_file)

    gt_df.columns = ['frame', 'id', 'x1', 'y1', 'w', 'h', 'score', 'x', 'y', 'z']
    pred_df.columns = ['frame', 'id', 'x1', 'y1', 'w', 'h', 'score', 'x', 'y', 'z']
    gt_df['centroid_x'] = gt_df['x1'] + gt_df['w'] / 2
    gt_df['centroid_y'] = gt_df['y1'] + gt_df['h'] / 2
    pred_df['centroid_x'] = pred_df['x1'] + pred_df['w'] / 2
    pred_df['centroid_y'] = pred_df['y1'] + pred_df['h'] / 2

    # Get the unique fish IDs in the ground truth and predicted data
    gt_fish_ids = gt_df['id'].unique()
    pred_fish_ids = pred_df['id'].unique()

    # Calculate the MSE for each fish from the GT
    mse = 0
    for fish_id in gt_fish_ids:
        gt_fish_df = gt_df[gt_df['id'] == fish_id]
        pred_fish_df = pred_df[pred_df['id'] == fish_id]

        # Only iterate over the frames where the fish is present in the GT
        for frame_num in gt_fish_df['frame'].unique():
            gt_fish_df_frame = gt_fish_df[gt_fish_df['frame'] == frame_num]
            pred_fish_df_frame = pred_fish_df[pred_fish_df['frame'] == frame_num]
            if pred_fish_df_frame.empty:
                # print(f'FISHNOTFOUND {frame_num} {gt_mot_file} {pred_mot_file}')
                mse += np.sqrt((gt_fish_df_frame['centroid_x'].values)**2 + (gt_fish_df_frame['centroid_y'].values)**2)
            else:
                mse += np.sqrt((gt_fish_df_frame['centroid_x'].values - pred_fish_df_frame['centroid_x'].values)**2 + (gt_fish_df_frame['centroid_y'].values - pred_fish_df_frame['centroid_y'].values)**2)

    return np.sqrt(mse)