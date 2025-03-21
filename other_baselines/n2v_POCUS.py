# Imports necessary to execute the code
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from careamics.lightning import (  
    create_careamics_module,
    create_predict_datamodule,
    create_train_datamodule,
)

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
import torch
from pytorch_lightning.strategies import DDPStrategy
import os

def custom_read_data(file_path: Path, *args: list, **kwargs: dict) -> np.ndarray:
    """
    Read a single frame from a file path
    """
    # print(f"Reading {file_path}")
    assert os.path.exists(file_path), f"File {file_path} does not exist"
    frame = cv2.imread(str(file_path))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.array(frame_gray)
    return frame


def main():
    DATA_DIR="/root/covid19_ultrasound/data"
    # DATA_DIR="/root/POCUS_data"
    RESULTS_DIR="/root/covid19_ultrasound"


    DEBUG = False
    # Set training data path
    FRAMES_DIR = f"{DATA_DIR}/image_dataset_baseline_flat_symlinks"
    root_path = Path(FRAMES_DIR)

    # create paths for the data
    data_path = Path(root_path)
    train_path = data_path
    val_path = Path(f"{DATA_DIR}/image_dataset_baseline_flat_symlinks") / "pneumonia_Pneu_liftl_pneu_case3_clip5.mp4_frame25.jpg"

    # Visualize the data
    single_frame = cv2.imread(str(val_path))
    # convert to nparray
    single_frame = np.array(single_frame)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(single_frame)
    ax.set_title("Training Image")
    plt.savefig("single_frame.png")

    # Create model
    model = create_careamics_module(algorithm="n2v", loss="n2v", architecture="UNet",)

    # Data module
    data = create_train_datamodule(
        train_data = train_path,
        val_data = val_path,
        data_type = "custom",
        patch_size = (128, 128),
        axes = "YX",
        batch_size = 32,
        val_percentage = 0.01,
        val_minimum_patches = 5,
        read_source_func = custom_read_data,
        # extension_filter = ".jpg",
    )
    # data.num_workers = 8
    data.prepare_data()
    # data.setup()

    results_path = f"{RESULTS_DIR}/n2v_POTUS_results" 

    # set callbacks
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
        save_last=True  # Saves the last model
    )


    if not DEBUG:
        wandb_logger = WandbLogger(
            project="SAVeD",  
            name="n2v_POTUS",            
        )
        logger = wandb_logger
    else:
        logger = None


    # create lightning trainer
    trainer = Trainer(
        max_epochs=1 if DEBUG else 120,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        default_root_dir=results_path,
        devices=1 if DEBUG else -1,
        callbacks=[checkpoint_callback_best, checkpoint_callback_last], 
        logger =  logger,
        strategy=DDPStrategy(find_unused_parameters=True),
    )

    # train
    trainer.fit(model, datamodule=data)

if __name__ == "__main__":
    main()