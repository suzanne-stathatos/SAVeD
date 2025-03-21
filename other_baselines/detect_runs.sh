# N2V
source n2v_env/bin/activate
# CFC22
python n2v_cfc22.py
python n2v_predict_cfc22.py
python setup/add_denoised_to_raw_2_ch.py\
    --denoised_imgs /root/Data/CFC22/n2v_cfc22_results \
    --raw_imgs /root/Data/CFC22/frames/raw_bs \
    --added_imgs /root/Data/CFC22/n2v_cfc22_2_ch


# POCUS
python n2v_POCUS.py
python n2v_predict_POCUS.py
python setup/add_denoised_to_raw_2_ch.py\
    --denoised_imgs /root/covid19_ultrasound/n2v_POTUS_results \
    --raw_imgs /root/covid19_ultrasound/data/image_dataset_baseline \
    --added_imgs /root/covid19_ultrasound/data/image_dataset_n2v_2_ch
# Move folders around to be in the right spot
python3 scripts/cross_val_splitter.py --splits 5
python3 scripts/train_covid19.py --data_dir ../data/cross_validation/ --fold 0 --epochs 40


# UDVD
source udvd_env/bin/activate
# CFC22
python udvd/fish_train.py --model blind-video-net-4 --channels 1 --out-channels 1 --loss mse --data-path /root/Data/CFC22/frames/raw_all --dataset CFC --batch-size 140 --image-size 128 --n-frames 5 --stride 64 --lr 1e-4 --num-epochs 10 --step-checkpoints
domains=("kenai-train" "kenai-val" "elwha")
for domain in "${domains[@]}"; do
    python fish_denoise.py --input_dir /root/Data/CFC22/frames/raw/$domain --output_dir /root/Data/CFC22/udvd_full_frames_denoised/$domain --model /root/other_baselines/experiments/udvd_full/checkpoints/checkpoint_best.pt
done
for domain in "${domains[@]}"; do
    python setup/add_denoised_to_raw_2_ch.py\
        --denoised_imgs /root/Data/CFC22/udvd_full_frames_denoised/$domain \
        --raw_imgs /root/Data/CFC22/frames/raw_bs/$domain \
        --added_imgs /mnt/8TBSSD/udvd_full_frames_denoised_with_bs_2_ch/$domain
done
python setup/convert_mot_to_yolo.py --mot-dir "/root/Data/CFC22/annotations_mot" --frames-dir "/root/Data/CFC22/udvd_full_frames_denoised_with_bs_2_ch" --output-dir "/root/Data/CFC22/yolo_dataset_udvd_full_with_bs_2_ch"
python train_detector.py --data '/root/Data/CFC22/yolo_dataset_udvd_full_with_bs_2_ch/data.yaml' --epochs 5 --batch-size 16 --weights 'yolov5s.pt' --img-size 896 --project 'yolo_train' --name 'udvd_full'
GTDIR="/root/Data/CFC22/annotations_mot"
domains=("elwha")
for domain in "${domains[@]}"; do
    python detect.py --dataset $domain --det_model yolo_train/udvd_full/weights/best.pt --input_dir /mnt/8TBSSD/udvd_full_frames_denoised_with_bs_2_ch --output_dir /mnt/8TBSSD/udvd_full_results --gt_dir $GTDIR
    python eval_detector_tm.py --dataset $domain --det_model yolo_train/udvd_full/weights/best.pt --input_dir /mnt/8TBSSD/udvd_full_frames_denoised_with_bs_2_ch --output_dir /mnt/8TBSSD/udvd_full_results --gt_dir $GTDIR
    python track.py --dataset $domain --frames_dir /mnt/8TBSSD/udvd_full_frames_denoised_with_bs_2_ch --detections_dir /mnt/8TBSSD/udvd_full_results
    python track_eval.py --dataset $domain --frames_dir /mnt/8TBSSD/udvd_full_frames_denoised_with_bs_2_ch --track_dir /mnt/8TBSSD/udvd_full_results --annos_root $GTDIR
    python count.py --dataset $domain --frames_dir /mnt/8TBSSD/udvd_full_frames_denoised_with_bs_2_ch --track_dir /mnt/8TBSSD/udvd_full_results
    python eval_count.py --dataset $domain --frames_dir /mnt/8TBSSD/udvd_full_frames_denoised_with_bs_2_ch --track_dir /mnt/8TBSSD/udvd_full_results | grep nMAE
done

# UDVD POCUS
# convert all videos to frames
python setup/convert_videos_to_frames.py --video-dir /root/covid19_ultrasound/data/pocus_videos/convex --output-dir /root/covid19_ultrasound/data/pocus_videos_frames/convex_baseline
python setup/convert_videos_to_frames.py --video-dir /root/covid19_ultrasound/data/pocus_videos/linear --output-dir /root/covid19_ultrasound/data/pocus_videos_frames/linear_baseline
python setup/create_dataset_with_all_frames.py --frames_dir /root/covid19_ultrasound/data/pocus_videos_frames --frames_output_dir /root/covid19_ultrasound/data/pocus_videos_frames_all

python udvd/fish_train.py --model blind-video-net-4 --channels 1 --out-channels 1 --loss mse --data-path /root/covid19_ultrasound/data/pocus_videos_frames_all --dataset CFC --batch-size 140 --image-size 128 --n-frames 5 --stride 64 --lr 1e-4 --num-epochs 10 --step-checkpoints
python other_baselines/udvd/notebook_demos/POCUS_denoise.py --input_dir /root/covid19_ultrasound/data/pocus_videos_frames/convex_baseline --output_dir /root/covid19_ultrasound/data/pocus_videos_frames/convex_udvd --model /root/udvd_POCUS_checkpoints_best.pt
python other_baselines/udvd/notebook_demos/POCUS_denoise.py --input_dir /root/covid19_ultrasound/data/pocus_videos_frames/linear_baseline --output_dir /root/covid19_ultrasound/data/pocus_videos_frames/linear_udvd --model /root/udvd_POCUS_checkpoints_best.pt

# convert frames folders BACK to videos
python setup/convert_frames_to_videos.py --frames_dir /root/covid19_ultrasound/data/pocus_videos_frames/convex_udvd --video_dir /root/covid19_ultrasound/data/pocus_videos/convex_udvd
python setup/convert_frames_to_videos.py --frames_dir /root/covid19_ultrasound/data/pocus_videos_frames/linear_udvd --video_dir /root/covid19_ultrasound/data/pocus_videos/linear_udvd

# Classical-filters POCUS
python setup/convert_videos_to_frames.py --video-dir /root/covid19_ultrasound/data/pocus_videos/convex --output-dir /root/covid19_ultrasound/data/pocus_videos_frames/convex_baseline
python setup/convert_videos_to_frames.py --video-dir /root/covid19_ultrasound/data/pocus_videos/linear --output-dir /root/covid19_ultrasound/data/pocus_videos_frames/linear_baseline

python setup/classical_filter_frames_POCUS.py --frames-dir /root/covid19_ultrasound/data/pocus_videos_frames --output-dir /root/covid19_ultrasound/data/pocus_videos_frames_median --filter-type median
python setup/convert_frames_to_videos.py --frames_dir /root/covid19_ultrasound/data/pocus_videos_frames_median/linear_baseline --video_dir /root/covid19_ultrasound/data/pocus_videos/linear_median
python setup/convert_frames_to_videos.py --frames_dir /root/covid19_ultrasound/data/pocus_videos_frames_median/convex_baseline --video_dir /root/covid19_ultrasound/data/pocus_videos/convex_median

python setup/classical_filter_frames_POCUS.py --frames-dir /root/covid19_ultrasound/data/pocus_videos_frames_other --output-dir /root/covid19_ultrasound/data/pocus_videos_frames_mean --filter-type mean
python setup/convert_frames_to_videos.py --frames_dir /root/covid19_ultrasound/data/pocus_videos_frames_mean/linear_baseline --video_dir /root/covid19_ultrasound/data/pocus_videos/linear_mean
python setup/convert_frames_to_videos.py --frames_dir /root/covid19_ultrasound/data/pocus_videos_frames_mean/convex_baseline --video_dir /root/covid19_ultrasound/data/pocus_videos/convex_mean

python setup/classical_filter_frames_POCUS.py --frames-dir /root/covid19_ultrasound/data/pocus_videos_frames_other --output-dir /root/covid19_ultrasound/data/pocus_videos_frames_gaussian --filter-type gaussian

# UDVD BUV
python setup/create_dataset_with_all_frames.py --frames_dir /root/CVA-Net/miccai_buv/rawframes --frames_output_dir /root/CVA-Net/miccai_buv/rawframes_all
python setup/create_dataset_with_all_frames.py --frames_dir /root/CVA-Net/miccai_buv/rawframes_mean --frames_output_dir /root/CVA-Net/miccai_buv/rawframes_mean_all
python udvd/fish_train.py --model blind-video-net-4 --channels 1 --out-channels 1 --loss mse --data-path /root/CVA-Net/miccai_buv/rawframes_all --dataset CFC --batch-size 64 --image-size 128 --n-frames 5 --stride 64 --lr 1e-4 --num-epochs 10 --step-checkpoints
# run the model over the frames
classes=("benign" "malignant")
for class in "${classes[@]}"; do
    python denoise_CVA.py --input_dir /root/CVA-Net/miccai_buv_baseline/rawframes/$class --output_dir /root/CVA-Net/miccai_buv/rawframes_udvd_denoised/$class --model /root/SAVeD/other_baselines/udvd_denoise_CVA_checkpoints/checkpoint_best.pt
done

# Classical-filters BUV
python setup/classical_filter_frames_CVA.py --frames-dir /root/CVA-Net/miccai_buv/rawframes --output-dir /root/CVA-Net/miccai_buv/rawframes_median --filter-type median
python setup/classical_filter_frames_CVA.py --frames-dir /root/CVA-Net/miccai_buv/rawframes --output-dir /root/CVA-Net/miccai_buv/rawframes_mean --filter-type mean
python setup/classical_filter_frames_CVA.py --frames-dir /root/CVA-Net/miccai_buv_baseline/rawframes --output-dir /root/CVA-Net/miccai_buv_mean/rawframes_mean --filter-type mean
python setup/classical_filter_frames_CVA.py --frames-dir /root/CVA-Net/miccai_buv_baseline/rawframes --output-dir /root/CVA-Net/miccai_buv_gaussian/rawframes_gaussian --filter-type gaussian