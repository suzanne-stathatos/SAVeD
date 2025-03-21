from n2v_POTUS import custom_read_data
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import functools
from careamics.lightning import create_careamics_module, create_predict_datamodule, create_train_datamodule
from careamics.prediction_utils import convert_outputs
from tqdm import tqdm
import numpy as np
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import os
from concurrent.futures import ThreadPoolExecutor
import cv2

def load_model(model_path):
    try: 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['state_dict']
        model = create_careamics_module(algorithm="n2v", loss="n2v", architecture="UNet")
        model.load_state_dict(state_dict)
        model = model.to(device)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

DATA_DIR="/root/CVA-Net-modified-to-work/miccai_buv"
FRAMES_DIR = f"/root/CVA-Net-modified-to-work/miccai_buv/rawframes_baseline_all_flat_symlinks"
VIDEO_DIR = f"/root/CVA-Net-modified-to-work/miccai_buv/rawframes_baseline"
root_path = Path(FRAMES_DIR)
data_path = Path(root_path)
train_path = data_path
val_path = Path(f"{DATA_DIR}/rawframes_baseline_all_flat_symlinks") / "benign_x28f299ceb056964c_000073.png"

data = create_train_datamodule(
    train_data=train_path,
    val_data=val_path,
    data_type="custom",
    patch_size=(128, 128),
    axes="YX",
    batch_size=32,
    val_percentage=0.01,
    val_minimum_patches=5,
    read_source_func=custom_read_data,
)

data.prepare_data()

RESULTS_DIR = "/root/CVA-Net-modified-to-work"
results_path = f"/root/CVA-Net-modified-to-work/miccai_buv/rawframes_n2v" 
model_path = f'{RESULTS_DIR}/n2v_CVA_results/best_model.ckpt'

print("Loading model...")
model = load_model(model_path)
print("Model loaded successfully")

checkpoint_callback_best = ModelCheckpoint(
    dirpath=f'{results_path}',
    filename='best_model',
    save_top_k=1,
    verbose=True,
    monitor='train_loss',
    mode='min'
)

checkpoint_callback_last = ModelCheckpoint(
    dirpath=f'{results_path}',
    filename='last_model',
    save_top_k=1,
    verbose=True,
    save_last=True
)

print("Setting up trainer...")
trainer = Trainer(
    max_epochs=0,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    default_root_dir=results_path,
    devices=-1,
    callbacks=[checkpoint_callback_best, checkpoint_callback_last], 
    logger=False,
    strategy=DDPStrategy(find_unused_parameters=True),
)
print("Trainer set up successfully")

def process_image(path):
    img = np.array(Image.open(path).convert('L')).astype(np.float32)
    return img.mean(), img.std(), img.size

def calculate_dataset_statistics_parallel(image_paths, num_workers=4):
    print("Calculating dataset statistics...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_image, image_paths), total=len(image_paths)))
    total_pixels = sum(n for _, _, n in results)
    weighted_mean = sum(m * n for m, _, n in results) / total_pixels
    weighted_std = np.sqrt(sum(((s**2 + m**2) * n) for m, s, n in results) / total_pixels - weighted_mean**2)
    print(f"\nDataset statistics:")
    print(f"Mean: {weighted_mean:.3f}")
    print(f"Std:  {weighted_std:.3f}")
    return weighted_mean, weighted_std

def get_image_paths(data_module):
    train_data_path = Path(data_module.train_data)
    if train_data_path.is_dir():
        image_paths = list(train_data_path.glob('*.tif'))
        image_paths.extend(train_data_path.glob('*.png'))
        image_paths.extend(train_data_path.glob('*.jpg'))
        return [str(path) for path in image_paths]
    else:
        raise ValueError(f"Train data path {train_data_path} is not a directory")

def save_prediction(pred, save_path):
    img_array = np.squeeze(pred)
    if img_array.dtype != np.uint8 and np.max(img_array) <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
    elif img_array.dtype != np.uint8 and np.max(img_array) > 1.0:
        img_array = img_array.astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save(save_path)

image_paths = get_image_paths(data)
print(f"Calculating dataset statistics for {len(image_paths)} images...")
means, stds = calculate_dataset_statistics_parallel(image_paths)
means = [means/255.]
stds = [stds/255.]
print(f"Means: {means}")
print(f"Stds: {stds}")
print("Dataset statistics calculated successfully")

def process_single_image(img_path, results_path, domain, clip, model, trainer, means, stds):
    predict_data = create_predict_datamodule(
        pred_data=img_path,
        data_type="custom",
        axes="YX",
        image_means=means,
        image_stds=stds,
        tile_size=(64, 64),
        tile_overlap=(2, 2),
        read_source_func=custom_read_data,
    )
    predict_data.prepare_data()
    predicted = trainer.predict(model, datamodule=predict_data)
    predicted_stitched = convert_outputs(predicted, tiled=True)
    img_result_path = Path(f"{results_path}/{domain}/{clip}/{img_path.name}")
    if isinstance(predicted_stitched, list):
        for i, pred in enumerate(predicted_stitched):
            save_prediction(pred, img_result_path)
    else:
        save_prediction(predicted_stitched, img_result_path)
    return img_result_path

def process_images_batch(domains, DATA_DIR, results_path, model, trainer, means, stds, batch_size=32):
    all_tasks = []
    for domain in domains:
        pred_path = Path(f"{DATA_DIR}/image_dataset_baseline/{domain}")
        for clip in os.listdir(pred_path):
            clip_path = Path(f"{pred_path}/{clip}")
            for img in os.listdir(clip_path):
                img_path = Path(f"{clip_path}/{img}")
                all_tasks.append((img_path, domain, clip))
    
    for i in range(0, len(all_tasks), batch_size):
        batch = all_tasks[i:i + batch_size]
        for img_path, domain, clip in batch:
            try:
                predict_data = create_predict_datamodule(
                    pred_data=img_path,
                    data_type="custom",
                    axes="YX",
                    image_means=means,
                    image_stds=stds,
                    tile_size=(128, 128),
                    tile_overlap=(2, 2),
                    read_source_func=custom_read_data,
                )
                predict_data.prepare_data()
                predicted = trainer.predict(model, datamodule=predict_data)
                predicted_stitched = convert_outputs(predicted, tiled=True)
                img_result_path = Path(f"{results_path}/{domain}/{clip}/{img_path.name}")
                img_result_path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(predicted_stitched, list):
                    for pred in predicted_stitched:
                        save_prediction(pred, img_result_path)
                else:
                    save_prediction(predicted_stitched, img_result_path)
                print(f"Processed image: {img_result_path}")
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")


def create_predict_datamodule_from_train(train_datamodule, pred_data_path, batch_size=32):
    return create_predict_datamodule(
        pred_data=pred_data_path,
        data_type="custom",
        axes="YX",
        image_means=means,
        image_stds=stds,
        tile_size=(128, 128),  
        tile_overlap=(2, 2),  # Adjust overlap as needed
        read_source_func=custom_read_data,
        batch_size=batch_size
    )

# # for images
# domains = ["covid", "pneumonia", "regular"]
# for domain in domains:
#     os.makedirs(f"{results_path}/{domain}", exist_ok=True)
# for domain in domains:
#     for clip in tqdm(os.listdir(Path(f"{DATA_DIR}/image_dataset_baseline/{domain}")), desc=f"Processing {domain}"):
#         pred_path = Path(f"{DATA_DIR}/image_dataset_baseline/{domain}/{clip}")
#         # Create a predict datamodule using the same configuration
#         predict_datamodule = create_predict_datamodule_from_train(data, pred_path, batch_size=32)

#         # Prepare the predict data
#         predict_datamodule.prepare_data()
#         predict_datamodule.setup()

#         # Process predictions in batches
#         # predict_dataloader = predict_datamodule.predict_dataloader()
#         # for batch in predict_dataloader:
#         predicted = trainer.predict(model, datamodule=predict_datamodule)
#         predicted_stitched = convert_outputs(predicted, tiled=True)
#         print(f'number of predictions: {len(predicted_stitched)}')
#         print(f'shape of first prediction: {predicted_stitched[0].shape}')
#         clip_result_path = Path(f"{results_path}/{domain}/{clip}")
#         os.makedirs(clip_result_path, exist_ok=True)
#         for i, pred in enumerate(predicted_stitched):
#             # Save predictions
#             img_result_path = Path(f"{clip_result_path}/{i}.jpg")
#             save_prediction(pred, img_result_path)
#         print(f"Saved images to {clip_result_path}")


def convert_video_to_frames(video_path):
    frames = [] 
    video_reader = cv2.VideoCapture(video_path)
    while True:
        ret, frame = video_reader.read()
        if not ret:
            break
        frames.append(frame)
    return frames

# for videos
domains = ['benign', 'malignant']
for domain in domains:
    videos_to_denoise = os.path.join(VIDEO_DIR, domain)
    for video in tqdm(os.listdir(videos_to_denoise), desc="Processing videos"):
        video_path = os.path.join(videos_to_denoise, video)
        if not os.path.isdir(video_path):
            continue
        # convert the video to a list of frames and save them to a folder
        frames = sorted(os.listdir(video_path))

        denoised_frames = []
        # verify that output directory exists
        os.makedirs(os.path.join(results_path, domain, video), exist_ok=True)

        # create a predict datamodule using the same configuration
        predict_datamodule = create_predict_datamodule_from_train(data, video_path, batch_size=32)
        # prepare the predict data
        predict_datamodule.prepare_data()
        predict_datamodule.setup()

        # Process the precitions in batches
        predicted = trainer.predict(model, datamodule=predict_datamodule)
        predicted_stitched = convert_outputs(predicted, tiled=True)
        print(f'number of predictions: {len(predicted_stitched)}')
        print(f'shape of first prediction: {predicted_stitched[0].shape}')
        print(f'Video path: {video_path}')
        print(f'len(preds): {len(predicted_stitched)}')
        for i, pred in enumerate(predicted_stitched):
            # Save predictions
            img_array = np.squeeze(pred)
            if img_array.dtype != np.uint8 and np.max(img_array) <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            elif img_array.dtype != np.uint8 and np.max(img_array) > 1.0:
                img_array = img_array.astype(np.uint8)
            img = Image.fromarray(img_array)
            denoised_frames.append(img)
        os.makedirs(os.path.join(results_path, domain, video), exist_ok=True)
        for i, frame in enumerate(denoised_frames):
            print(os.path.join(results_path, domain, video, f"{i:06d}.png"))
            frame.save(os.path.join(results_path, domain, video, f"{i:06d}.png"))