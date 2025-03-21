FRAMES_DIR="/root/Data/CFC22/frames"
ANNOTATIONS_DIR="/root/Data/CFC22"
domains=("kenai-train" "kenai-val" "elwha")
OUTDIR="/root/Data/CFC22"

# Vary bottleneck size

# all_denoised_1024_fine_bnck_16
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 32 --resolution_target 1024 --fine_layers --bottleneck_size 16 --target "l1" --results_path "checkpoints/all_denoised_1024_fine_bnck_16" --model_type "cnn"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_16/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/4TBSSD/all_denoised_1024_fine_bnck_16/$domain --model_type "cnn" --fine_layers --bottleneck_size 16 --resolution_size 1024
done

# all_denoised_1024_fine_bnck_32
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 32 --resolution_target 1024 --fine_layers --bottleneck_size 32 --target "l1" --results_path "checkpoints/all_denoised_1024_fine_bnck_32" --model_type "cnn"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_32/best_model-v1.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $FRAMES_DIR/all_denoised_1024_fine_bnck_32/$domain --model_type "cnn" --fine_layers --bottleneck_size 32 --resolution_size 1024
done

# all_denoised_1024_fine_bnck_64
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 32 --resolution_target 1024 --fine_layers --bottleneck_size 64 --target "l1" --results_path "checkpoints/all_denoised_1024_fine_bnck_64" --model_type "cnn"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_64/best_model-v1.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $FRAMES_DIR/all_denoised_1024_fine_bnck_64/$domain --model_type "cnn" --fine_layers --bottleneck_size 64 --resolution_size 1024
done

# all_denoised_1024_fine_bnck_128
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 32 --resolution_target 1024 --fine_layers --bottleneck_size 128 --target "l1" --results_path "checkpoints/all_denoised_1024_fine_bnck_128" --model_type "cnn"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_128/best_model-v1.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/4TBSSD/all_denoised_1024_fine_bnck_128/$domain --model_type "cnn" --fine_layers --bottleneck_size 128 --resolution_size 1024
done

# all_denoised_1024_fine_bnck_256
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 32 --resolution_target 1024 --fine_layers --bottleneck_size 256 --target "l1" --results_path "checkpoints/all_denoised_1024_fine_bnck_256" --model_type "cnn"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_256/best_model-v1.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/4TBSSD/all_denoised_1024_fine_bnck_256/$domain --model_type "cnn" --fine_layers --bottleneck_size 256 --resolution_size 1024
done

# all_denoised_1024_fine_bnck_512
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 32 --resolution_target 1024 --fine_layers --bottleneck_size 512 --target "l1" --results_path "checkpoints/all_denoised_1024_fine_bnck_512" --model_type "cnn"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512/best_model-v1.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/4TBSSD/all_denoised_1024_fine_bnck_512/$domain --model_type "cnn" --fine_layers --bottleneck_size 512 --resolution_size 1024
done


# 2048-resolution
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_2048_fine_bnck_32/best_model-v1.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $OUTDIR/frames/all_denoised_2048_fine_bnck_32/$domain --model_type "cnn" --fine_layers --bottleneck_size 32 --resolution_size 2048
done

for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_2048_fine_bnck_64/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $FRAMES_DIR/all_denoised_2048_fine_bnck_64/$domain --model_type "cnn" --fine_layers --bottleneck_size 64 --resolution_size 2048
done

for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_2048_fine_bnck_128/best_model-v1.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $FRAMES_DIR/all_denoised_2048_fine_bnck_128/$domain --model_type "cnn" --fine_layers --bottleneck_size 128 --resolution_size 2048
done

for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_2048_fine_bnck_256/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $FRAMES_DIR/all_denoised_2048_fine_bnck_256/$domain --model_type "cnn" --fine_layers --bottleneck_size 256 --resolution_size 2048
done

for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_2048_fine_bnck_512/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $FRAMES_DIR/all_denoised_2048_fine_bnck_512/$domain --model_type "cnn" --fine_layers --bottleneck_size 512 --resolution_size 2048
done

for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_2048_fine_bnck_1024/best_model-v1.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $FRAMES_DIR/all_denoised_2048_fine_bnck_1024/$domain --model_type "cnn" --fine_layers --bottleneck_size 1024 --resolution_size 2048
done


# Vary loss functions

# all_denoised_1024_fine_bnck_512_with_l1_loss
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 32 --resolution_target 1024 --fine_layers --bottleneck_size 512 --target "l1" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_with_l1_loss" --model_type "cnn" --loss "l1"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_with_l1_loss/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $OUTDIR/frames/all_denoised_1024_fine_bnck_512_with_l1_loss/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512
done

# all_denoised_1024_fine_bnck_positive_diff_and_curr_frame_with_l1_loss
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 32 --resolution_target 1024 --fine_layers --bottleneck_size 512 --target "positive_diff_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_positive_diff_and_curr_frame_with_l1_loss" --model_type "cnn" --loss "l1"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_positive_diff_and_curr_frame_with_l1_loss/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $OUTDIR/frames/all_denoised_1024_fine_bnck_512_positive_diff_and_curr_frame_with_l1_loss/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512
done

# Vary reconstruction target

# all_denoised_1024_fine_bnck_512_positive_diff_and_curr_frame
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 10 --batch_size 32 --resolution_target 1024 --fine_layers --bottleneck_size 512 --target "positive_diff_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_positive_diff_and_curr_frame" --model_type "cnn"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_positive_diff_and_curr_frame/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $OUTDIR/all_denoised_1024_fine_bnck_512_positive_diff_and_curr_frame/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512
done

# all_denoised_1024_fine_bnck_512_positive_diff_and_curr_frame_bs_in
echo "all_denoised_1024_fine_bnck_512_positive_diff_and_curr_frame_bs_in"
# make background subtracted frames
python setup/background_subtract_frames.py --frames-dir $FRAMES_DIR/raw --output-dir $FRAMES_DIR/raw_bs
python setup/create_dataset_with_all_frames.py --frames_dir $FRAMES_DIR/raw_bs --annotations_dir $ANNOTATIONS_DIR/annotations_mot --frames_output_dir $FRAMES_DIR/raw_bs_all --annotations_output_dir $ANNOTATIONS_DIR/annotations_mot_all
python main.py --base_folder $FRAMES_DIR/raw_bs_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 32 --resolution_target 1024 --fine_layers --bottleneck_size 512 --target "positive_diff_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_positive_diff_and_curr_frame_bs_in" --model_type "cnn"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_positive_diff_and_curr_frame_bs_in/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $OUTDIR/all_denoised_1024_fine_bnck_512_positive_diff_and_curr_frame_bs_in/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512
done

# all_denoised_1024_fine_bnck_512_perceptual_loss
echo "all_denoised_1024_fine_bnck_512_perceptual_loss"
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 10 --batch_size 8 --resolution_target 1024 --fine_layers --bottleneck_size 512 --target "l1" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_perceptual_loss" --model_type "cnn" --loss "perceptual"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_perceptual_loss/best_model-v1.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $OUTDIR/all_denoised_1024_fine_bnck_512_perceptual_loss/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512
done

# all_denoised_1024_fine_bnck_512_stddev_all
echo "all_denoised_1024_fine_bnck_512_stddev_all"
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 32 --resolution_target 1024 --fine_layers --bottleneck_size 512 --target "stddev_all" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_stddev_all" --model_type "cnn"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_stddev_all/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $OUTDIR/all_denoised_1024_fine_bnck_512_stddev_all/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512
done

# all_denoised_2048_fine_bnck_32
echo "all_denoised_2048_fine_bnck_32"
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 8 --resolution_target 2048 --fine_layers --bottleneck_size 32 --target "l1" --results_path "checkpoints/all_denoised_2048_fine_bnck_32" --model_type "cnn"  

# all_denoised_2048_fine_bnck_64
echo "all_denoised_2048_fine_bnck_64"
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 8 --resolution_target 2048 --fine_layers --bottleneck_size 64 --target "l1" --results_path "checkpoints/all_denoised_2048_fine_bnck_64" --model_type "cnn"

# all_denoised_2048_fine_bnck_128
echo "all_denoised_2048_fine_bnck_128"
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 8 --resolution_target 2048 --fine_layers --bottleneck_size 128 --target "l1" --results_path "checkpoints/all_denoised_2048_fine_bnck_128" --model_type "cnn"

# all_denoised_2048_fine_bnck_256
echo "all_denoised_2048_fine_bnck_256"
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 8 --resolution_target 2048 --fine_layers --bottleneck_size 256 --target "l1" --results_path "checkpoints/all_denoised_2048_fine_bnck_256" --model_type "cnn"

# all_denoised_2048_fine_bnck_512
echo "all_denoised_2048_fine_bnck_512"
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 8 --resolution_target 2048 --fine_layers --bottleneck_size 512 --target "l1" --results_path "checkpoints/all_denoised_2048_fine_bnck_512" --model_type "cnn"

# all_denoised_2048_fine_bnck_1024
echo "all_denoised_2048_fine_bnck_1024"
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 8 --resolution_target 2048 --fine_layers --bottleneck_size 1024 --target "l1" --results_path "checkpoints/all_denoised_2048_fine_bnck_1024" --model_type "cnn"


# Autoencoder 
# AE_1024_identity
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 32 --target "identity" --results_path "checkpoints/all_denoised_ae_1024_identity_t-1" --model_type "ae" --resolution_target 1024
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_ae_1024_identity_t-1/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/denoised_all_ae_1024_identity_t-1/$domain --model_type "ae" --resolution_size 1024
done

# UNet downscaled
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 10 --batch_size 8 --target "l1" --results_path "checkpoints/all_denoised_unet_downscaled" --model_type "unet_downscaled"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_unet_downscaled/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $FRAMES_DIR/denoised_all_unet_downscaled/$domain --model_type "unet_downscaled"
done

# unet_1024_identity
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 10 --batch_size 2 --target "identity" --results_path "checkpoints/all_denoised_unet_identity" --model_type "unet" --resolution_target 1024
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_unet_identity/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/denoised_all_unet_identity/$domain --model_type "unet" --resolution_size 1024
done

# unet_downscaled_1024_identity
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 13 --batch_size 4 --target "identity" --results_path "checkpoints/all_denoised_unet_identity_downscaled" --model_type "unet_downscaled" --resolution_target 1024 --resume
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_unet_identity_downscaled/last_model-v1.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/denoised_all_unet_identity_downscaled/$domain --model_type "unet_downscaled" --resolution_size 1024
done

# unet_1024_gap1
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 10 --batch_size 4 --target "gap1_and_curr_frame" --results_path "checkpoints/all_denoised_unet_1024_gap1" --model_type "unet" --resolution_target 1024
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_unet_1024_gap1/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/denoised_all_unet_1024_gap1/$domain --model_type "unet" --resolution_size 1024
done

# unet_downscaled_1024_gap1
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 4 --target "gap1_and_curr_frame" --results_path "checkpoints/all_denoised_unet_1024_gap1_downscaled" --model_type "unet_downscaled" --resolution_target 1024
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_unet_1024_gap1_downscaled/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/denoised_all_unet_1024_gap1_downscaled/$domain --model_type "unet_downscaled" --resolution_size 1024
done

# unet_sum_minus_5b
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 10 --batch_size 2 --target "sum_minus_5_background" --results_path "checkpoints/all_denoised_unet_sum_minus_5b" --model_type "unet" --resolution_target 1024
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_unet_sum_minus_5b/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/denoised_all_unet_sum_minus_5b/$domain --model_type "unet" --resolution_size 1024
done

# unet_downscaled_sum_minus_5b
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 2 --target "sum_minus_5_background" --results_path "checkpoints/all_denoised_unet_sum_minus_5b_downscaled" --model_type "unet_downscaled" --resolution_target 1024
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_unet_sum_minus_5b_downscaled/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/denoised_all_unet_sum_minus_5b_downscaled/$domain --model_type "unet_downscaled" --resolution_size 1024
done

# unet_positive_diff_and_curr_frame
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 10 --batch_size 8 --target "positive_diff_and_curr_frame" --results_path "checkpoints/all_denoised_unet_positive_diff_and_curr_frame" --model_type "unet"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_unet_positive_diff_and_curr_frame/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $FRAMES_DIR/all_denoised_unet_positive_diff_and_curr_frame/$domain --model_type "unet"
done

# unet_positive_diff_last
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_unet_positive_diff_and_curr_frame/last_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $FRAMES_DIR/all_denoised_unet_positive_diff_and_curr_frame_last/$domain --model_type "unet"
done


# unet_positive_diff_and_curr_frame_downscaled
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 10 --batch_size 8 --target "positive_diff_and_curr_frame" --results_path "checkpoints/all_denoised_unet_positive_diff_and_curr_frame_downscaled" --model_type "unet_downscaled"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_unet_positive_diff_and_curr_frame_downscaled/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $FRAMES_DIR/denoised_all_unet_positive_diff_and_curr_frame_downscaled/$domain --model_type "unet_downscaled"
done

# UNet_bs_in_identity
python main.py --base_folder $FRAMES_DIR/raw_bs_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_bs_all --epochs 10 --batch_size 8 --target "identity" --results_path "checkpoints/all_denoised_unet_bs_in_identity" --model_type "unet"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_unet_bs_in_identity/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $FRAMES_DIR/all_denoised_unet_bs_in_identity/$domain --model_type "unet"
done

# unet_positive_diff_bs_in
python main.py --base_folder $FRAMES_DIR/raw_bs_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_bs_all --epochs 10 --batch_size 8 --target "positive_diff_and_curr_frame" --results_path "checkpoints/all_denoised_unet_bs_in_positive_diff_and_curr_frame" --model_type "unet"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_unet_bs_in_positive_diff_and_curr_frame/best_model.ckpt" --input_dir $FRAMES_DIR/raw_bs/$domain --output_dir $FRAMES_DIR/denoised_all_unet_bs_in_positive_diff_and_curr_frame/$domain --model_type "unet"
done

# Unet3D
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 10 --batch_size 4 --resolution_target 1024 --target "gap1_and_curr_frame" --results_path "checkpoints/unet_4hlayers_3d" --model_type "unet_4hlayers_3d"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/unet_4hlayers_3d/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/unet_4hlayers_3d/$domain --model_type "unet_4hlayers_3d" --resolution_size 1024
done


# Augmentations on all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP
# salt_and_pepper_p = 0.25
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --target "gap1_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_snp_0.25" --model_type "cnn" --salt_and_pepper_p 0.25 --with_skip_connections
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_snp_0.25/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_snp_0.25/$domain --model_type "cnn" --fine_layers --bottleneck_size 512 --resolution_size 1024 --with_skip_connections
done
# salt_and_pepper_p = 0.5
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --target "gap1_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_snp_0.5" --model_type "cnn" --salt_and_pepper_p 0.5 --with_skip_connections
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_snp_0.5/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_snp_0.5/$domain --model_type "cnn" --fine_layers --bottleneck_size 512 --resolution_size 1024 --with_skip_connections
done
# salt_and_pepper_p = 0.75
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --target "gap1_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_snp_0.75" --model_type "cnn" --salt_and_pepper_p 0.75 --with_skip_connections
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_snp_0.75/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_snp_0.75/$domain --model_type "cnn" --fine_layers --bottleneck_size 512 --resolution_size 1024 --with_skip_connections
done

# gaussian_blur_p = 0.25
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --target "gap1_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_gb_0.25" --model_type "cnn" --gaussian_blur_p 0.25 --with_skip_connections
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_gb_0.25/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $OUTDIR/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_gb_0.25/$domain --model_type "cnn" --fine_layers --bottleneck_size 512 --resolution_size 1024 --with_skip_connections
done
# gaussian_blur_p = 0.5
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --target "gap1_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_gb_0.5" --model_type "cnn" --gaussian_blur_p 0.5 --with_skip_connections
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_gb_0.5/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $OUTDIR/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_gb_0.5/$domain --model_type "cnn" --fine_layers --bottleneck_size 512 --resolution_size 1024 --with_skip_connections
done
# gaussian_blur_p = 0.75
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --target "gap1_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_gb_0.75" --model_type "cnn" --gaussian_blur_p 0.75 --with_skip_connections
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_gb_0.75/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $OUTDIR/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_gb_0.75/$domain --model_type "cnn" --fine_layers --bottleneck_size 512 --resolution_size 1024 --with_skip_connections
done

# motion_blur_p = 0.25
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --target "gap1_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_mb_0.25" --model_type "cnn" --motion_blur_p 0.25 --with_skip_connections
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_mb_0.25/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $OUTDIR/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_mb_0.25/$domain --model_type "cnn" --fine_layers --bottleneck_size 512 --resolution_size 1024 --with_skip_connections
done
# motion_blur_p = 0.5
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --target "gap1_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_mb_0.5" --model_type "cnn" --motion_blur_p 0.5 --with_skip_connections
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_mb_0.5/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $OUTDIR/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_mb_0.5/$domain --model_type "cnn" --fine_layers --bottleneck_size 512 --resolution_size 1024 --with_skip_connections
done
# motion_blur_p = 0.75
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --target "gap1_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_mb_0.75" --model_type "cnn" --motion_blur_p 0.75 --with_skip_connections
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_mb_0.75/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $OUTDIR/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_mb_0.75/$domain --model_type "cnn" --fine_layers --bottleneck_size 512 --resolution_size 1024 --with_skip_connections
done

# brightness_p = 0.25
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --target "gap1_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_brightness_0.25" --model_type "cnn" --brightness_p 0.25 --with_skip_connections
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_brightness_0.25/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $OUTDIR/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_brightness_0.25/$domain --model_type "cnn" --fine_layers --bottleneck_size 512 --resolution_size 1024 --with_skip_connections
done
# brightness_p = 0.5
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --target "gap1_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_brightness_0.5" --model_type "cnn" --brightness_p 0.5 --with_skip_connections
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_brightness_0.5/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $OUTDIR/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_brightness_0.5/$domain --model_type "cnn" --fine_layers --bottleneck_size 512 --resolution_size 1024 --with_skip_connections
done
# brightness_p = 0.75
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --target "gap1_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_brightness_0.75" --model_type "cnn" --brightness_p 0.75 --with_skip_connections
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_brightness_0.75/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $OUTDIR/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_brightness_0.75/$domain --model_type "cnn" --fine_layers --bottleneck_size 512 --resolution_size 1024 --with_skip_connections
done

# erase_p = 0.25
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --target "gap1_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_erase_0.25" --model_type "cnn" --erase_p 0.25 --with_skip_connections
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_erase_0.25/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $OUTDIR/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_erase_0.25/$domain --model_type "cnn" --fine_layers --bottleneck_size 512 --resolution_size 1024 --with_skip_connections
done
# erase_p = 0.5
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --target "gap1_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_erase_0.5" --model_type "cnn" --erase_p 0.5 --with_skip_connections
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_erase_0.5/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $OUTDIR/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_erase_0.5/$domain --model_type "cnn" --fine_layers --bottleneck_size 512 --resolution_size 1024 --with_skip_connections
done
# erase_p = 0.75
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --target "gap1_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_erase_0.75" --model_type "cnn" --erase_p 0.75 --with_skip_connections
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_erase_0.75/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $OUTDIR/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_erase_0.75/$domain --model_type "cnn" --fine_layers --bottleneck_size 512 --resolution_size 1024 --with_skip_connections
done

# Background-subtracted frames as input

# Bs in Bs out
python setup/create_dataset_with_all_frames.py --frames_dir $FRAMES_DIR/raw_bs --annotations_dir $ANNOTATIONS_DIR/annotations_mot --frames_output_dir $FRAMES_DIR/raw_bs_all --annotations_output_dir $ANNOTATIONS_DIR/annotations_mot_bs_all
python main.py --base_folder $FRAMES_DIR/raw_bs_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_bs_all --epochs 20 --batch_size 16 --fine_layers --resolution_target 1024 --bottleneck_size 512 --target "bs_global" --results_path "checkpoints/denoised_all_bs_in_bs_out_1024_512_fine"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/denoised_all_bs_in_bs_out_1024_512_fine/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/4TBSSD/all_denoised_bs_in_bs_out_1024_512_fine/$domain --model_type "cnn" --fine_layers --resolution_size 1024 --bottleneck_size 512
done

# bs in identity
python main.py --base_folder $FRAMES_DIR/raw_bs_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all/ --epochs 10 --batch_size 32 --fine_layers --resolution_target 1024 --bottleneck_size 512 --target "identity" --results_path "checkpoints/denoised_all_bs_in_identity_1024_512_fine"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/denoised_all_bs_in_identity_1024_512_fine/best_model.ckpt" --input_dir $FRAMES_DIR/raw_bs/$domain --output_dir $FRAMES_DIR/denoised_all_bs_in_identity_1024_512_fine/$domain --model_type "cnn" --fine_layers --resolution_size 1024 --bottleneck_size 512
done

# Skip connections

# all_denoised_1024_fine_bnck_512_SKIP
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --target "l1" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_SKIP" --model_type "cnn" --with_skip_connections
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_SKIP/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $FRAMES_DIR/all_denoised_1024_fine_bnck_512_SKIP/$domain --model_type "cnn" --fine_layers --bottleneck_size 512 --resolution_size 1024 --with_skip_connections
done

# all_denoised_1024_fine_bnck_512_SKIP_positive_diff_and_curr_frame
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --target "positive_diff_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_SKIP_positive_diff_and_curr_frame" --model_type "cnn" --with_skip_connections
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_SKIP_positive_diff_and_curr_frame/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $FRAMES_DIR/all_denoised_1024_fine_bnck_512_SKIP_positive_diff_and_curr_frame/$domain --model_type "cnn" --fine_layers --bottleneck_size 512 --resolution_size 1024 --with_skip_connections
done

# Larger Data
# all_denoised_2048_fine_bnck_512_SKIP_positive_diff_and_curr_frame
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 4 --resolution_target 2048 --fine_layers --bottleneck_size 512 --target "positive_diff_and_curr_frame" --results_path "checkpoints/all_denoised_2048_fine_bnck_512_SKIP_positive_diff_and_curr_frame" --model_type "cnn" --with_skip_connections
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_2048_fine_bnck_512_SKIP_positive_diff_and_curr_frame/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir $FRAMES_DIR/all_denoised_2048_fine_bnck_512_SKIP_positive_diff_and_curr_frame/$domain --model_type "cnn" --fine_layers --bottleneck_size 512 --resolution_size 2048 --with_skip_connections
done

# Different targets

# all_denoised_1024_fine_bnck_512_sum_minus_3_background
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 32 --resolution_target 1024 --fine_layers --bottleneck_size 512 --target "sum_minus_3_background" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_sum_minus_3_background" --model_type "cnn"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_sum_minus_3_background/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/all_denoised_1024_fine_bnck_512_sum_minus_3_background/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512
done

# all_denoised_1024_fine_bnck_512_sum_minus_3_background_SKIP
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --target "sum_minus_3_background" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_sum_minus_3_background_SKIP" --model_type "cnn"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_sum_minus_3_background_SKIP/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/all_denoised_1024_fine_bnck_512_sum_minus_3_background_SKIP/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections
done

# all_denoised_1024_fine_bnck_512_sum_minus_5_background
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 32 --resolution_target 1024 --fine_layers --bottleneck_size 512 --target "sum_minus_5_background" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_sum_minus_5_background" --model_type "cnn"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_sum_minus_5_background/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/all_denoised_1024_fine_bnck_512_sum_minus_5_background/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512
done

# all_denoised_1024_fine_bnck_512_sum_minus_5_background_SKIP
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --target "sum_minus_5_background" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_sum_minus_5_background_SKIP" --model_type "cnn"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_sum_minus_5_background_SKIP/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/all_denoised_1024_fine_bnck_512_sum_minus_5_background_SKIP/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections
done

# all_denoised_1024_fine_bnck_512_gap2_and_curr_frame
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 32 --resolution_target 1024 --fine_layers --bottleneck_size 512 --target "gap2_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_gap2_and_curr_frame" --model_type "cnn"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_gap2_and_curr_frame/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/all_denoised_1024_fine_bnck_512_gap2_and_curr_frame/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512
done

# all_denoised_1024_fine_bnck_512_SKIP_gap2
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --target "gap2_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_SKIP_gap2" --model_type "cnn"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_SKIP_gap2/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/all_denoised_1024_fine_bnck_512_SKIP_gap2/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections
done

# all_denoised_1024_fine_bnck_512_gap1_and_curr_frame
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 32 --resolution_target 1024 --fine_layers --bottleneck_size 512 --target "gap1_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame" --model_type "cnn"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512
done

# all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --target "gap1_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP" --model_type "cnn"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections
done

# Residual skips and resnet blocks

# 1024_512_SKIP_gap1_and_curr_frame_with_residual_skips
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --with_residual_connections --target "gap1_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_residual" --model_type "cnn"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_residual/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_residual/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --with_residual_connections
done

# 1024_512_SKIP_stddev_all_with_residual_skips
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --with_residual_connections --target "stddev_all" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_stddev_all_skip_residual" --model_type "cnn"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_stddev_all_skip_residual/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/all_denoised_1024_fine_bnck_512_stddev_all_skip_residual/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --with_residual_connections
done

# 1024_512_SKIP_with_resnet_blocks
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 8 --resolution_target 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --with_resnet_blocks --target "gap1_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_resnet" --model_type "cnn"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_resnet/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_resnet/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --with_resnet_blocks
done

# 1024_512_SKIP_with_resnet_blocks_stddev_all
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 8 --resolution_target 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --with_resnet_blocks --target "stddev_all" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_stddev_all_skip_resnet" --model_type "cnn"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_stddev_all_skip_resnet/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/all_denoised_1024_fine_bnck_512_stddev_all_skip_resnet/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --with_resnet_blocks
done

# Optimizers

# all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_sgd
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --target "gap1_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_sgd" --model_type "cnn" --optimizer "sgd"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_sgd/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_sgd/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections
done

# all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_rmsprop
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --target "gap1_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_rmsprop" --model_type "cnn" --optimizer "rmsprop"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_rmsprop/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_rmsprop/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections
done

# Schedulers

# all_denoised_1024_fine_bnck_512_SKIP_gap1_adamw_plateau_scheduler_f0.1_pat5
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --target "gap1_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_adamw_plateau_scheduler_f0.1_pat5" --model_type "cnn" --optimizer "adamw" --scheduler "plateau" --factor 0.1 --patience 5
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_adamw_plateau_scheduler_f0.1_pat5/best_model-v1.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_adamw_plateau_scheduler_f0.1_pat5/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections
done

# all_denoised_1024_fine_bnck_512_SKIP_gap1_adamw_plateau_scheduler_f0.1_pat2
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --target "gap1_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_adamw_plateau_scheduler_f0.1_pat2" --model_type "cnn" --optimizer "adamw" --scheduler "plateau" --factor 0.1 --patience 2
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_adamw_plateau_scheduler_f0.1_pat2/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_adamw_plateau_scheduler_f0.1_pat2/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections
done

# all_denoised_1024_fine_bnck_512_SKIP_gap1_adamw_plateau_scheduler_f0.5_pat5
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --target "gap1_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_adamw_plateau_scheduler_f0.5_pat5" --model_type "cnn" --optimizer "adamw" --scheduler "plateau" --factor 0.5 --patience 5
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_adamw_plateau_scheduler_f0.5_pat5/best_model-v1.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_adamw_plateau_scheduler_f0.5_pat5/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections
done

# all_denoised_1024_fine_bnck_512_SKIP_gap1_adamw_plateau_scheduler_f0.5_pat2
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --target "gap1_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_adamw_plateau_scheduler_f0.5_pat2" --model_type "cnn" --optimizer "adamw" --scheduler "plateau" --factor 0.5 --patience 2
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_adamw_plateau_scheduler_f0.5_pat2/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_adamw_plateau_scheduler_f0.5_pat2/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections
done

# all_denoised_1024_fine_bnck_512_SKIP_gap1_adamw_plateau_scheduler_f0.05_pat2
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --target "gap1_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_adamw_plateau_scheduler_f0.05_pat2" --model_type "cnn" --optimizer "adamw" --scheduler "plateau" --factor 0.05 --patience 2
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_adamw_plateau_scheduler_f0.05_pat2/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_adamw_plateau_scheduler_f0.05_pat2/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections
done

# all_denoised_1024_fine_bnck_512_SKIP_gap1_adamw_stepLR_ss4_gamma0.1
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --target "gap1_and_curr_frame" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_adamw_stepLR_ss4_gamma0.1" --model_type "cnn" --optimizer "adamw" --scheduler "step" --step_size 4 --gamma 0.1
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_adamw_stepLR_ss4_gamma0.1/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/all_denoised_1024_fine_bnck_512_gap1_and_curr_frame_skip_adamw_stepLR_ss4_gamma0.1/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections
done

# all_denoised_1024_fine_bnck_512_identity
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 32 --resolution_target 1024 --fine_layers --bottleneck_size 512 --target "identity" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_identity" --model_type "cnn"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_identity/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/all_denoised_1024_fine_bnck_512_identity/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512
done

# all_denoised_1024_fine_bnck_512_identity_SKIP
python main.py --base_folder $FRAMES_DIR/raw_all --annotations_folder $ANNOTATIONS_DIR/annotations_mot_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --target "identity" --results_path "checkpoints/all_denoised_1024_fine_bnck_512_identity_SKIP" --model_type "cnn"
for domain in "${domains[@]}"; do
    python denoise.py --model "checkpoints/all_denoised_1024_fine_bnck_512_identity_SKIP/best_model.ckpt" --input_dir $FRAMES_DIR/raw/$domain --output_dir /mnt/8TBSSD/frames/all_denoised_1024_fine_bnck_512_identity_SKIP/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections
done


### MEDICAL
# Breast Lesions
classes=("benign" "malignant")


python main.py --dataset "CVA" --base_folder /root/CVA-Net/miccai_buv/rawframes_median --epochs 40 --batch_size 8 --resolution_target 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --target "gap1_inverted_and_curr_frame" --results_path "checkpoints/CVA_denoised_1024_fine_bnck_512_gap1_inverted_and_curr_frame_SKIP_40_epoch_median_sq" --model_type "cnn"

python setup/create_dataset_with_all_frames.py --frames_dir /root/CVA-Net/miccai_buv_mean/rawframes_mean --frames_output_dir /root/CVA-Net/miccai_buv_mean/rawframes_mean_all
python main.py --dataset "CVA" --base_folder /root/CVA-Net/miccai_buv_mean/rawframes_mean_all --epochs 40 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --target "gap1_inverted_and_curr_frame" --results_path "checkpoints/CVA_denoised_1024_fine_bnck_512_gap1_inverted_and_curr_frame_SKIP_40_epoch_meansq" --model_type "cnn"

# Unet3D
python main.py --dataset "CVA" --base_folder /root/CVA-Net/miccai_buv_mean/rawframes_mean_all --epochs 10 --batch_size 4 --resolution_target 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --target "gap1_inverted_and_curr_frame" --results_path "checkpoints/CVA_denoised_1024_fine_bnck_512_gap1_inverted_and_curr_frame_SKIP_40_epoch_meansq_ae" --model_type "unet_tiny_3d"
for class in "${classes[@]}"; do
    python denoise_CVA.py --model "checkpoints/CVA_denoised_1024_fine_bnck_512_gap1_inverted_and_curr_frame_SKIP_40_epoch_meansq_unet3d/best_model.ckpt" --input_dir /root/CVA-Net/miccai_buv_baseline/rawframes/$class --output_dir /root/CVA-Net/miccai_buv/rawframes_denoised_unet3d/$class --model_type "unet_tiny_3d" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections
done
for class in "${classes[@]}"; do
    python setup/add_denoised_to_raw_2_ch.py \
        --denoised_imgs /root/CVA-Net/miccai_buv/rawframes_denoised_unet3d/$class \
        --raw_imgs /root/CVA-Net/miccai_buv_baseline/rawframes/$class \
        --added_imgs /root/CVA-Net/miccai_buv/rawframes_denoised_unet3d_2_ch/$class
done


# UNet
python setup/create_dataset_with_all_frames.py --frames_dir /root/CVA-Net/miccai_buv_baseline/rawframes --frames_output_dir /root/CVA-Net/miccai_buv_baseline/rawframes_all
python main.py --dataset "CVA" --base_folder /root/CVA-Net/miccai_buv_baseline/rawframes_all --epochs 10 --batch_size 2 --resolution_target 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --target "gap1_inverted_and_curr_frame" --results_path "checkpoints/CVA_denoised_1024_fine_bnck_512_gap1_inverted_and_curr_frame_SKIP_40_epoch_meansq_unet" --model_type "unet"
for class in "${classes[@]}"; do
    python denoise_CVA.py --model "checkpoints/CVA_denoised_1024_fine_bnck_512_gap1_inverted_and_curr_frame_SKIP_40_epoch_meansq_unet/best_model.ckpt" --input_dir /root/CVA-Net/miccai_buv_baseline/rawframes/$class --output_dir /root/CVA-Net/miccai_buv_baseline/rawframes_denoised/$class --model_type "unet" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections
done
for class in "${classes[@]}"; do
    python setup/add_denoised_to_raw_2_ch.py \
        --denoised_imgs /root/CVA-Net/miccai_buv_baseline/rawframes_denoised/$class \
        --raw_imgs /root/CVA-Net/miccai_buv_baseline/rawframes/$class \
        --added_imgs /root/CVA-Net/miccai_buv/rawframes_denoised_2_ch_unet/$class
done

# AE
python setup/create_dataset_with_all_frames.py --frames_dir /root/CVA-Net/miccai_buv_baseline/rawframes --frames_output_dir /root/CVA-Net/miccai_buv_baseline/rawframes_all
python main.py --dataset "CVA" --base_folder /root/CVA-Net/miccai_buv_baseline/rawframes_all --epochs 40 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --target "gap1_inverted_and_curr_frame" --results_path "checkpoints/CVA_denoised_1024_fine_bnck_512_gap1_inverted_and_curr_frame_SKIP_40_epoch_meansq_ae" --model_type "ae"
for class in "${classes[@]}"; do
    python denoise_CVA.py --model "checkpoints/CVA_denoised_1024_fine_bnck_512_gap1_inverted_and_curr_frame_SKIP_40_epoch_meansq_ae/best_model.ckpt" --input_dir /root/CVA-Net/miccai_buv_baseline/rawframes/$class --output_dir /root/CVA-Net/miccai_buv_baseline/rawframes_denoised/$class --model_type "ae" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections
done
for class in "${classes[@]}"; do
    python setup/add_denoised_to_raw_2_ch.py \
        --denoised_imgs /root/CVA-Net/miccai_buv_baseline/rawframes_denoised/$class \
        --raw_imgs /root/CVA-Net/miccai_buv_baseline/rawframes/$class \
        --added_imgs /root/CVA-Net/miccai_buv_baseline/rawframes_denoised_2_ch_ae/$class
done

# median-denoised-sq
for class in "${classes[@]}"; do
    python denoise_CVA.py --model "checkpoints/CVA_denoised_1024_fine_bnck_512_gap1_inverted_and_curr_frame_SKIP_40_epochsq/best_model.ckpt" --input_dir /root/CVA-Net/miccai_buv/rawframes_median/$class --output_dir /root/CVA-Net/miccai_buv/rawframes_median_sq_denoised/$class --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections
done
for class in "${classes[@]}"; do    
    python setup/add_denoised_to_raw_2_ch.py\
        --denoised_imgs /root/CVA-Net/miccai_buv/rawframes_median_sq_denoised/$class \
        --raw_imgs /root/CVA-Net/miccai_buv/rawframes_median/$class \
        --added_imgs /root/CVA-Net/miccai_buv/rawframes_median_sq_denoised_2_ch/$class
done


# mean-denoised-sq
for class in "${classes[@]}"; do
    python denoise_CVA.py --model "checkpoints/CVA_denoised_1024_fine_bnck_512_gap1_inverted_and_curr_frame_SKIP_40_epoch_meansq/best_model.ckpt" --input_dir /root/CVA-Net/miccai_buv_mean/rawframes_mean/$class --output_dir /root/CVA-Net/miccai_buv_mean/rawframes_mean_sq_denoised/$class --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections
done
for class in "${classes[@]}"; do    
    python setup/add_denoised_to_raw_2_ch.py\
        --denoised_imgs /root/CVA-Net/miccai_buv_mean/rawframes_mean_sq_denoised/$class \
        --raw_imgs /root/CVA-Net/miccai_buv_mean/rawframes_mean/$class \
        --added_imgs /root/CVA-Net/miccai_buv/rawframes_mean_sq_denoised_2_ch/$class
done


# median-denoised
for class in "${classes[@]}"; do
    python denoise_CVA.py --model "checkpoints/CVA_denoised_1024_fine_bnck_512_gap1_inverted_and_curr_frame_SKIP_40_epoch/best_model.ckpt" --input_dir /root/CVA-Net/miccai_buv_median/rawframes_median/$class --output_dir /root/CVA-Net/miccai_buv_median/rawframes_median_denoised/$class --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections
done
for class in "${classes[@]}"; do    
    python setup/add_denoised_to_raw_2_ch.py\
        --denoised_imgs /root/CVA-Net/miccai_buv_median/rawframes_median_denoised/$class \
        --raw_imgs /root/CVA-Net/miccai_buv_median/rawframes_median/$class \
        --added_imgs /root/CVA-Net/miccai_buv/rawframes_median_denoised_2_ch/$class
done


# mean-denoised
for class in "${classes[@]}"; do
    python denoise_CVA.py --model "checkpoints/CVA_denoised_1024_fine_bnck_512_gap1_inverted_and_curr_frame_SKIP_40_epoch_mean/best_model.ckpt" --input_dir /root/CVA-Net/miccai_buv_mean/rawframes_mean/$class --output_dir /root/CVA-Net/miccai_buv_mean/rawframes_mean_denoised/$class --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections
done
for class in "${classes[@]}"; do
    python denoise_CVA.py --model "checkpoints/CVA_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP/best_model.ckpt" --input_dir /home/suzanne/CVA-Net/miccai_buv/rawframes/$class --output_dir /home/suzanne/CVA-Net/miccai_buv/rawframes_all_denoised/$class --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections
done
for class in "${classes[@]}"; do
    python setup/add_denoised_to_raw_2_ch.py \
        --denoised_imgs /root/CVA-Net/miccai_buv_mean/rawframes_mean_denoised/$class \
        --raw_imgs /root/CVA-Net/miccai_buv_mean/rawframes_mean/$class \
        --added_imgs /root/CVA-Net/miccai_buv/rawframes_mean_denoised_2_ch/$class
done


#inverted to accentuate the dark parts rather than the bright parts
python main.py --dataset "CVA" --base_folder /home/suzanne/CVA-Net/miccai_buv/rawframes_all --epochs 20 --batch_size 16 --resolution_target 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --target "gap1_inverted_and_curr_frame" --results_path "checkpoints/CVA_denoised_1024_fine_bnck_512_gap1_inverted_and_curr_frame_SKIP" --model_type "cnn"
for class in "${classes[@]}"; do
    python denoise_CVA.py --model "checkpoints/CVA_denoised_1024_fine_bnck_512_gap1_inverted_and_curr_frame_SKIP/best_model.ckpt" --input_dir /home/suzanne/CVA-Net/miccai_buv/rawframes/$class --output_dir /home/suzanne/CVA-Net/miccai_buv/rawframes_all_inverted_denoised/$class --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections
done
for class in "${classes[@]}"; do
    python setup/add_denoised_to_raw_2_ch.py \
        --denoised_imgs /home/suzanne/CVA-Net/miccai_buv/rawframes_all_inverted_denoised/$class \
        --raw_imgs /home/suzanne/CVA-Net/miccai_buv/rawframes/$class \
        --added_imgs /home/suzanne/CVA-Net/miccai_buv/rawframes_all_inverted_denoised_2_ch/$class
done



# Lung ultrasounds
python setup/create_dataset_with_all_frames.py --frames_dir /root/covid19_ultrasound/data/pocus_videos_frames_median --frames_output_dir /root/covid19_ultrasound/data/pocus_videos_frames_median_flat
python main.py --dataset "POCUS" --base_folder /root/covid19_ultrasound/data/pocus_videos_frames_median_flat --epochs 10 --batch_size 2 --resolution_target 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --target "gap1_and_curr_frame" --results_path "checkpoints/POCUS_denoised_unet_gap1_and_curr_frame_SKIP_on_median" --model_type "unet"
python main.py --dataset "POCUS" --base_folder /root/covid19_ultrasound/data/pocus_videos_frames_median_flat --epochs 10 --batch_size 4 --resolution_target 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --target "gap1_and_curr_frame" --results_path "checkpoints/POCUS_denoised_unet_gap1_and_curr_frame_SKIP_on_median" --model_type "unet_tiny_3d"

# CNN
domains=("convex" "linear")
for domain in "${domains[@]}"; do
    python denoise_POCUS.py --model "checkpoints/POCUS_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP_on_median/best_model.ckpt" --input_dir /root/covid19_ultrasound/data/pocus_videos_frames_median/$domain --output_dir /root/covid19_ultrasound/data/pocus_videos_frames_median_denoised/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections
done
for domain in "${domains[@]}"; do
    python setup/add_denoised_to_raw_2_ch.py \
        --denoised_imgs /root/covid19_ultrasound/data/pocus_videos_frames_median_denoised/$domain \
        --raw_imgs /root/covid19_ultrasound/data/pocus_videos_frames_median/$domain \
        --added_imgs /root/covid19_ultrasound/data/pocus_videos_frames_median_denoised_2_ch/$domain
done
for domain in "${domains[@]}"; do
    python setup/convert_frames_to_videos.py --frames_dir /root/covid19_ultrasound/data/pocus_videos_frames_median_denoised_2_ch/$domain --video_dir "/root/covid19_ultrasound/data/pocus_videos/${domain}_median_denoised_2_ch"
done
for domain in "${domains[@]}"; do
    rm -rf /root/covid19_ultrasound/data/pocus_videos/${domain}
    mv /root/covid19_ultrasound/data/pocus_videos/${domain}_median_denoised_2_ch /root/covid19_ultrasound/data/pocus_videos/${domain}
done


# AE
domains=("convex" "linear")
for domain in "${domains[@]}"; do
    python denoise_POCUS.py --model "checkpoints/POCUS_denoised_ae_gap1_and_curr_frame_SKIP_on_median/best_model-v1.ckpt" --input_dir /root/covid19_ultrasound/data/pocus_videos_frames_median/$domain --output_dir /root/covid19_ultrasound/data/pocus_videos_frames_median_denoised_ae/$domain --model_type "ae" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections
done
for domain in "${domains[@]}"; do
    python setup/add_denoised_to_raw_2_ch.py \
        --denoised_imgs /root/covid19_ultrasound/data/pocus_videos_frames_median_denoised_ae/$domain \
        --raw_imgs /root/covid19_ultrasound/data/pocus_videos_frames_median/$domain \
        --added_imgs /root/covid19_ultrasound/data/pocus_videos_frames_median_denoised_ae_2_ch/$domain
done
for domain in "${domains[@]}"; do
    python setup/convert_frames_to_videos.py --frames_dir /root/covid19_ultrasound/data/pocus_videos_frames_median_denoised_ae_2_ch/$domain --video_dir "/root/covid19_ultrasound/data/pocus_videos/${domain}_median_denoised_ae_2_ch"
done
for domain in "${domains[@]}"; do
    mv /root/covid19_ultrasound/data/pocus_videos/${domain}_median_denoised_ae_2_ch /root/covid19_ultrasound/data/pocus_videos/${domain}
done


# Unet3D
python main.py --dataset "POCUS" --base_folder /root/covid19_ultrasound/data/pocus_videos_frames_median_flat --epochs 10 --batch_size 4 --resolution_target 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --target "gap1_and_curr_frame" --results_path "checkpoints/POCUS_denoised_unet_gap1_and_curr_frame_SKIP_on_median" --model_type "unet_tiny_3d"
domains=("convex" "linear")
for domain in "${domains[@]}"; do
    python denoise_POCUS.py --model "checkpoints/POCUS_denoised_unet_gap1_and_curr_frame_SKIP_on_median/best_model.ckpt" --input_dir /root/covid19_ultrasound/data/pocus_videos_frames_median/$domain --output_dir /root/covid19_ultrasound/data/pocus_videos_frames_median_denoised_unet3d/$domain --model_type "unet_tiny_3d" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections
done
for domain in "${domains[@]}"; do
    python setup/add_denoised_to_raw_2_ch.py \
        --denoised_imgs /root/covid19_ultrasound/data/pocus_videos_frames_median_denoised_unet3d/$domain \
        --raw_imgs /root/covid19_ultrasound/data/pocus_videos_frames_median/$domain \
        --added_imgs /root/covid19_ultrasound/data/pocus_videos_frames_median_denoised_unet3d_2_ch/$domain
done
for domain in "${domains[@]}"; do
    python setup/convert_frames_to_videos.py --frames_dir /root/covid19_ultrasound/data/pocus_videos_frames_median_denoised_unet3d_2_ch/$domain --video_dir "/root/covid19_ultrasound/data/pocus_videos/${domain}_denoised_unet3d_2_ch"
done
mkdir -p /root/covid19_ultrasound/data/pocus_videos_old
for domain in "${domains[@]}"; do
    mv /root/covid19_ultrasound/data/pocus_videos/${domain} /root/covid19_ultrasound/data/pocus_videos_old/${domain}
    mv /root/covid19_ultrasound/data/pocus_videos/${domain}_denoised_unet3d_2_ch /root/covid19_ultrasound/data/pocus_videos/${domain}
done


# Unet 
python main.py --dataset "POCUS" --base_folder /root/covid19_ultrasound/data/pocus_videos_frames_median_flat --epochs 10 --batch_size 2 --resolution_target 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --target "gap1_and_curr_frame" --results_path "checkpoints/POCUS_denoised_unet_gap1_and_curr_frame_SKIP_on_median" --model_type "unet"
domains=("convex" "linear")
for domain in "${domains[@]}"; do
    python denoise_POCUS.py --model "checkpoints/POCUS_denoised_unet_gap1_and_curr_frame_SKIP_on_median/best_model.ckpt" --input_dir /root/covid19_ultrasound/data/pocus_videos_frames_median/$domain --output_dir /root/covid19_ultrasound/data/pocus_videos_frames_median_denoised_unet/$domain --model_type "unet" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections
done
for domain in "${domains[@]}"; do
    python setup/add_denoised_to_raw_2_ch.py \
        --denoised_imgs /root/covid19_ultrasound/data/pocus_videos_frames_median_denoised_unet/$domain \
        --raw_imgs /root/covid19_ultrasound/data/pocus_videos_frames_median/$domain \
        --added_imgs /root/covid19_ultrasound/data/pocus_videos_frames_median_denoised_unet_2_ch/$domain
done
for domain in "${domains[@]}"; do
    python setup/convert_frames_to_videos.py --frames_dir /root/covid19_ultrasound/data/pocus_videos_frames_median_denoised_unet_2_ch/$domain --video_dir "/root/covid19_ultrasound/data/pocus_videos/${domain}_denoised_unet_2_ch"
done
for domain in "${domains[@]}"; do
    mv /root/covid19_ultrasound/data/pocus_videos_old/${domain}_denoised_unet_2_ch /root/covid19_ultrasound/data/pocus_videos/${domain}
done

# python denoise_POCUS.py --model "checkpoints/POCUS_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP/best_model.ckpt" --input_dir /root/POCUS_data/image_dataset --output_dir /root/POCUS_data/image_dataset_denoised --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections
# python denoise_POCUS_videos.py --model "checkpoints/POCUS_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP/best_model.ckpt" --input_dir /root/covid19_ultrasound/data/pocus_videos/convex --output_dir /root/covid19_ultrasound/data/pocus_videos/convex_denoised --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections
# mv /root/covid19_ultrasound/data/pocus_videos/convex /root/covid19_ultrasound/data/pocus_videos/convex_baseline
# mv /root/covid19_ultrasound/data/pocus_videos/convex_denoised /root/covid19_ultrasound/data/pocus_videos/convex

# python denoise_POCUS_videos.py --model "checkpoints/POCUS_denoised_1024_fine_bnck_512_gap1_and_curr_frame_SKIP/best_model.ckpt" --input_dir /root/covid19_ultrasound/data/pocus_videos/convex_baseline --output_dir /root/covid19_ultrasound/data/pocus_videos/convex --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections

# # count the number of frames in the dataset
# find /root/POCUS_data/image_dataset -type f | wc -l
# find /root/POCUS_data/denoised_frames -type f | wc -l


# Flurorescence Miscroscopy

python setup/convert_tiff_to_jpg.py --input_dir /mnt/8TBSSD/Fluo/Fluo/train --output_dir /mnt/8TBSSD/Fluo/Fluo_jpg/train --quality 100
python setup/convert_tiff_to_jpg.py --input_dir /mnt/8TBSSD/Fluo/Fluo/test --output_dir /mnt/8TBSSD/Fluo/Fluo_jpg/test --quality 100
python setup/create_dataset_with_all_frames.py --frames_dir /root/Fluo --frames_output_dir /root/Fluo_flat

python setup/classical_filter_frames_Fluo.py --frames-dir /root/Fluo --output-dir /root/Fluo_median --filter-type median
python setup/create_dataset_with_all_frames.py --frames_dir /root/Fluo_median --frames_output_dir /root/Fluo_median_flat
python setup/classical_filter_frames_Fluo.py --frames-dir /root/Fluo --output-dir /root/Fluo_mean --filter-type mean
python setup/create_dataset_with_all_frames.py --frames_dir /root/Fluo_mean --frames_output_dir /root/Fluo_mean_flat

python main.py --dataset "Fluo" --base_folder /root/Fluo_median_flat --epochs 1000 --batch_size 8 --resolution_target 1024 --fine_layers --bottleneck_size 512 --with_skip_connections --target "identity" --results_path "checkpoints/Fluo_median_denoised_1024sq_fine_bnck_512_identity_SKIP_1000_epochs" --model_type "cnn"

domains=("train" "test")
for domain in "${domains[@]}"; do
    python denoise_Fluo.py --model "checkpoints/Fluo_median_denoised_1024sq_fine_bnck_512_identity_SKIP_1000_epochs/best_model.ckpt" --input_dir /root/Fluo_median/$domain --output_dir /root/Fluo_median_denoised_identity_1000_epochssq/$domain --model_type "cnn" --resolution_size 1024 --fine_layers --bottleneck_size 512 --with_skip_connections
done

# combine denoised frames with original frames in 2_channels
for domain in "${domains[@]}"; do
    python setup/add_denoised_to_raw_2_ch.py \
        --denoised_imgs /root/Fluo_denoised_identity_1000_epochs/$domain \
        --raw_imgs /root/Fluo/$domain \
        --added_imgs /root/Fluo_denoised_identity_1000_epochs_2_ch/$domain
done

