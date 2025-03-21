'''
Usage:
python track_eval.py --dataset $domain --frames_dir /path/to/Data/CFC22/frames/oracle_bbox_masks --track_dir /path/to/oracle_bbox_masks_results
'''

import os
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
from TrackEval import trackeval
from utils.eval_utils import get_motion_from_mot, get_size_from_mot, get_trajectory_mse, get_avg_conf_from_mot, get_trajectory_l2_no_fp_penalty

# Directly set np.float and np.int to np.float64 and np.int64
np.float = np.float64
np.int = np.int64


def get_tracker_config ():
    quiet = False
    tracker_cfg = trackeval.Evaluator.get_default_eval_config()
    tracker_cfg['DISPLAY_LESS_PROGRESS'] = True
    tracker_cfg['TIME_PROGRESS'] = False
    tracker_cfg['USE_PARALLEL'] = False
    tracker_cfg['NUM_PARALLEL_CORES'] = 1
    tracker_cfg['PRINT_ONLY_COMBINED'] = not quiet
    tracker_cfg['PRINT_RESULTS'] = not quiet
    tracker_cfg['PRINT_CONFIG'] = not quiet
    return tracker_cfg

def run_eval ( args, datasets, metrics, metrics_dict ):
    tracker_cfg = get_tracker_config ( )

    evaluator = trackeval.Evaluator( tracker_cfg )
    metrics_dict['PRINT_CONFIG'] = tracker_cfg['PRINT_CONFIG']

    # Prints overall results
    results, messages = evaluator.evaluate ( datasets, metrics)

    # Print sequence-wise results
    print("Sequence-wise results also written to file")
    
    for dataset_name, dataset_results in results.items():
        print(f"Dataset: {dataset_name}")
        eval_path = os.path.join(args.track_save_path, 'metrics')
        os.makedirs(eval_path, exist_ok=True)

        print(f"Dataset: {dataset_name}")
        for tracker, tracker_results in dataset_results.items():
            MSE_scores = []
            print(f"Tracker: {tracker}")
            domain_tracker_scores_f = os.path.join(eval_path, f'{tracker}.txt')
            domain_tracker_scores_df = pd.DataFrame(columns=['Sequence', 'MOTA', 'IDF1', 'L2', 'true_fish_motion_score', 'pred_fish_motion_score', 'true_fish_size'] + [f'HOTA{i}' for i in range(19)]) # 19 different thresholds ranging from 0.05 to 0.95 in 0.05 increments, 19th is the overall average

            for sequence_name, sequence_results in tracker_results.items():
                if sequence_name != 'COMBINED_SEQ':
                    hota_score_all_alphas = sequence_results['pedestrian']['HOTA']['HOTA'] 
                    # first hota score - minimal localization threshold
                    hota_score_min_localization = hota_score_all_alphas[0]
                    mota_score = sequence_results['pedestrian']['CLEAR']['MOTA']
                    idf1_score = sequence_results['pedestrian']['Identity']['IDF1']
                    
                    # save all clip-level scores to a dataframe
                    gt_mot_file = os.path.join(args.annos_root, tracker, sequence_name, 'gt.txt')
                    pred_mot_file = os.path.join(args.track_save_path, 'mot_results_trackeval_format', tracker, 'preds', f"{sequence_name}.txt")
                    true_fish_motion_score = get_motion_from_mot(gt_mot_file)
                    pred_fish_motion_score = get_motion_from_mot(pred_mot_file)
                    true_fish_size = get_size_from_mot(gt_mot_file)
                    pred_fish_avg_conf = get_avg_conf_from_mot(pred_mot_file)
                    trajectory_l2_score = get_trajectory_l2_no_fp_penalty(gt_mot_file, pred_mot_file)
                    MSE_scores.append(trajectory_l2_score)

                    new_row = pd.DataFrame({
                        'Sequence': [sequence_name],
                        'HOTA': [hota_score_min_localization],
                        'MOTA': [mota_score],
                        'IDF1': [idf1_score],
                        'L2': [trajectory_l2_score],
                        'true_fish_motion_score': [true_fish_motion_score],
                        'pred_fish_motion_score': [pred_fish_motion_score],
                        'true_fish_size': [true_fish_size],
                        'pred_fish_avg_conf': [pred_fish_avg_conf],
                        **{f'HOTA{i}': [hota_score_all_alphas[i]] for i in range(18)}
                    })
                    domain_tracker_scores_df = pd.concat([domain_tracker_scores_df, new_row], ignore_index=True)
            domain_tracker_scores_df.to_csv(domain_tracker_scores_f, index=False)
            print(f"ALL scores for {tracker} on {dataset_name} saved to {domain_tracker_scores_f}")
            mean_hota = np.mean(domain_tracker_scores_df['HOTA'])
            mean_mota = np.mean(domain_tracker_scores_df['MOTA'])
            mean_idf1 = np.mean(domain_tracker_scores_df['IDF1'])
            mean_l2 = np.mean(domain_tracker_scores_df['L2'])
            mean_true_fish_motion_score = np.mean(domain_tracker_scores_df['true_fish_motion_score'])
            mean_pred_fish_motion_score = np.mean(domain_tracker_scores_df['pred_fish_motion_score'])
            mean_true_fish_size = np.mean(domain_tracker_scores_df['true_fish_size'])
            mean_pred_fish_avg_conf = np.mean(domain_tracker_scores_df['pred_fish_avg_conf'])
            print(f'Mean MOTA: {mean_mota:.4f}')
            print(f'Mean IDF1: {mean_idf1:.4f}')
            print(f'Mean HOTA: {mean_hota:.4f}')
            print(f'Mean true fish motion score: {mean_true_fish_motion_score:.4f}')
            print(f'Mean pred fish motion score: {mean_pred_fish_motion_score:.4f}')
            print(f'Mean true fish size: {mean_true_fish_size:.4f}')
            print(f'Mean pred fish avg conf: {mean_pred_fish_avg_conf:.4f}')            


def get_seq_len ( images_dir ):
    """Get sequence information of each clip from the images that the detector ran over"""
    # get unique sequence names
    all_jpg_files = [file for file in os.listdir ( images_dir ) if file.endswith('.jpg')]
    return len(all_jpg_files)    

def load_dataset_in_motchallenge_format(args):
    '''
    Load dataset in MOTChallenge format
    '''
    # Get domain name
    domain = args.dataset

    # Get the frames directory
    frames_dir = os.path.join(args.frames_dir, domain)

    # Get the root directories for the ground truth and predicted tracker results
    root_gt_dir = os.path.join(args.annos_root, domain)
    root_tracker_dir = os.path.join(args.track_save_path, 'mot_results_trackeval_format', domain, 'preds')
    seq_info = dict()

    # Iterate over each sequence in the ground truth and predicted tracker results
    for seq in tqdm(os.listdir(root_gt_dir)):
        seq_gt_dir = os.path.join(root_gt_dir, seq)
        seq_tracker_file = os.path.join(root_tracker_dir, f"{seq}.txt")

        if not os.path.exists(seq_tracker_file):
            print(f"Tracker file {seq_tracker_file} does not exist. Skipping.")
            continue

        # Get the sequence information for the current sequence
        seq_info[seq] = get_seq_len(os.path.join(frames_dir, seq))
    
    print(f'Track save path: {args.track_save_path}')
    dataset = trackeval.datasets.MotChallenge2DBox( {'SEQ_INFO': seq_info, 
                                                        'SPLIT_TO_EVAL': domain,
                                                        'GT_FOLDER': root_gt_dir,
                                                        'TRACKERS_FOLDER': os.path.join(args.track_save_path, 'mot_results_trackeval_format'),
                                                        'TRACKERS_TO_EVAL': [domain],
                                                        'OUTPUT_FOLDER': None, # same as trackers_folder
                                                        'TRACKER_SUB_FOLDER': 'preds',
                                                        'SKIP_SPLIT_FOL': True,
                                                        'TRACKER_LOC_FORMAT': '{trackers_folder}/{tracker}/preds/{seq}.txt',  # Specify exact format
                                                        'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt.txt',
                                                    'DO_PREPROC': False,
                                                    'PRINT_CONFIG': True
                                                    } )

    return dataset


def copy_pred_tracks_files(args):
    """Copy tracks.txt files to preds folder in the tracker results directory"""
    mot_results_dir = os.path.join(args.track_save_path, 'mot_results_trackeval_format')
    os.makedirs(mot_results_dir, exist_ok=True)
    print(f'Moving tracks.txt files to preds folder in {mot_results_dir}')

    tracks_for_domain = os.path.join(args.track_dir, args.dataset)
    track_preds_dir = os.path.join(mot_results_dir, args.dataset, 'preds')
    if os.path.exists(track_preds_dir):
        shutil.rmtree(track_preds_dir)
    os.makedirs(track_preds_dir, exist_ok=True)

    for seq in os.listdir(tracks_for_domain):
        seq_dir = os.path.join(tracks_for_domain, seq)
        pred = os.path.join(seq_dir, 'tracks.txt')
        if os.path.exists(pred):
            shutil.copy(pred, os.path.join(track_preds_dir, f'{seq}.txt'))
            # print(f"Copied {pred} to {os.path.join(track_preds_dir, f'{seq}.txt')}")
        else:
            print(f'File {pred} does not exist')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset name', required=True)
    parser.add_argument('--frames_dir', type=str, help='Frames directory', required=True)
    parser.add_argument('--track_dir', type=str, help='Tracker results directory', required=True)
    parser.add_argument('--annos_root', type=str, help='Root directory for ground truth annotations', required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args.track_save_path = args.track_dir
    
    # Need to copy pred.txt files to preds folder in the tracker results directory
    copy_pred_tracks_files(args)

    dataset = load_dataset_in_motchallenge_format ( args )
    
    # specify which metrics to compute
    # These should likely go in tracker_cfg.yaml
    metrics_dict = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    metrics_list = [trackeval.metrics.HOTA(metrics_dict), trackeval.metrics.CLEAR(metrics_dict), trackeval.metrics.Identity(metrics_dict)] 
    print(f"Evaluating {dataset}")
    run_eval ( args, [dataset], metrics_list, metrics_dict )