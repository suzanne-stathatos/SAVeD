import os
import argparse
import torch
from torchvision import transforms
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from dataset.NoisyVideoDataModule import NoisyVideoDataModule
from dataset.NoisyVideoDataModule_prev_frames_5 import NoisyVideoDataModule_prev_frames_5
from dataset.NoisyVideoDataModuleWithWindowSize import NoisyVideoDataModuleWithWindowSize
from dataset.NoisyVideoDataModuleWithWindowSizePOCUS import NoisyVideoDataModuleWithWindowSizePOCUS
from dataset.NoisyVideoDataModuleWithWindowSizeCVA import NoisyVideoDataModuleWithWindowSizeCVA
from dataset.CVADataModule import CVADataModule
from dataset.POCUSDataModule import POCUSDataModule
from dataset.FluoDataModule import FluoDataModule
from dataset.Augmentations import Augmentation
from model.DenoiserLightning import DenoiserLightning
from model.DenoiserLightningAlt import DenoiserLightningAlt
from model.DenoiserLightning_prev_frames_5 import DenoiserLightning_prev_frames_5
from model.DenoisingCNN import DenoisingCNN_512, DenoisingCNN_1024, DenoisingCNN_2048, DenoisingCNN_2048_with_skips, DenoisingCNN_512_with_skips, DenoisingCNN_1024_with_skips, DenoisingCNN_1024_with_residual_connections, DenoisingCNN_1024_with_skips_and_resnet_blocks
from model.AE import AE_1024
from model.UNet import UNet
import importlib
import sys
from model.UNet_downscaled import UNet_downscaled
importlib.reload(sys.modules['model.UNet_downscaled'])
from model.UNetTiny3D import UNetTiny3D
from model.DenoisingCNN_prev_frames_5 import DenoisingCNN_512_prev_frames_5, DenoisingCNN_1024_prev_frames_5

torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser(description="Train a denoising CNN")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--model_type", type=str, default="cnn", choices=["cnn", "unet", "unet_downscaled", "unet_tiny_3d", "ae", "resnet50", "vit"])
    parser.add_argument("--base_folder", type=str, required=True)
    parser.add_argument("--annotations_folder", type=str, required=True)
    parser.add_argument("--inference_clip1", type=str, default="kenai-channel_2018-08-16-JD228_Channel_Stratum1_Set1_CH_2018-08-16_060006_532_732")
    parser.add_argument("--inference_clip2", type=str, default="elwha_Elwha_2018_OM_ARIS_2018_07_21_2018-07-21_200000_4780_5231") 
    parser.add_argument("--inference_clip3", type=str, default="elwha_Elwha_2018_OM_ARIS_2018_07_14_2018-07-14_020000_2830_3281") 
    parser.add_argument("--inference_clip4", type=str, default="elwha_Elwha_2018_OM_ARIS_2018_09_14_2018-09-14_060000_1844_2345") 
    parser.add_argument("--inference_clip5", type=str, default="elwha_Elwha_2018_OM_ARIS_2018_07_10_2018-07-10_100000_2628_3079")
    parser.add_argument("--results_path", type=str, default="checkpoints/tmp")
    parser.add_argument("--resolution_target", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_prev_frames", type=int, default=2)
    parser.add_argument("--target", type=str, default="l1")
    parser.add_argument("--fine_layers", action="store_true")
    parser.add_argument("--bottleneck_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--loss", type=str, default="mse")
    parser.add_argument("--loss_lambda", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    parser.add_argument("--with_skip_connections", action="store_true", help="Use skip connections")
    parser.add_argument("--with_residual_connections", action="store_true", help="Use residual connections")
    parser.add_argument("--with_resnet_blocks", action="store_true", help="Use resnet blocks")
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--salt_and_pepper_p", type=float, default=0.0)
    parser.add_argument("--gaussian_blur_p", type=float, default=0.0)
    parser.add_argument("--motion_blur_p", type=float, default=0.0)
    parser.add_argument("--brightness_p", type=float, default=0.0)
    parser.add_argument("--erase_p", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--scheduler", type=str, default="step")
    parser.add_argument("--step_size", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--factor", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dataset", type=str, default="CFC22")
    parser.add_argument("--no_logging", action="store_true")
    return parser.parse_args()

args = parse_args()
DEBUG = False

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize((args.resolution_target, args.resolution_target//2)),
    transforms.ToTensor()
])
if args.model_type == "unet_tiny_3d":
    args.window_size = 31
    # do not resize for unet_tiny_3d
    transform = transforms.Compose([
        transforms.Resize((args.resolution_target, args.resolution_target//2)),
        transforms.ToTensor()
    ])

if DEBUG:
    args.batch_size = 1

# Parameters
batch_size = args.batch_size
learning_rate = 0.0005
epochs = 1 if DEBUG else args.epochs

if len(args.annotations_folder) == 0:
    args.annotations_folder = None

# Initialize the data modules
if args.dataset == "CVA":
    transform = transforms.Compose([
        transforms.Resize((args.resolution_target, args.resolution_target)),
        transforms.ToTensor()
    ])
    data_module = CVADataModule(data_dir=args.base_folder, 
                                transform=transform, 
                                target=args.target,
                                batch_size=args.batch_size, 
                                num_workers=args.batch_size,
                                pin_memory=True,
                                debug=DEBUG,
                                debug_size=10)
elif args.dataset == "Fluo":
    transform = transforms.Compose([
        transforms.Resize((args.resolution_target, args.resolution_target)),
        transforms.ToTensor()
    ])
    data_module = FluoDataModule(data_dir=args.base_folder, 
                                transform=transform, 
                                target=args.target,
                                batch_size=args.batch_size, 
                                num_workers=args.batch_size,
                                pin_memory=True,
                                debug=DEBUG,
                                debug_size=10)
elif args.dataset == "POCUS":
    data_module = POCUSDataModule(data_dir=args.base_folder, 
                                transform=transform, 
                                target=args.target,
                                batch_size=args.batch_size, 
                                num_workers=args.batch_size,
                                pin_memory=True,
                                debug=DEBUG,
                                debug_size=10)
elif args.dataset == "CFC22" and args.num_prev_frames == 2:
    data_module = NoisyVideoDataModule(data_dir=args.base_folder, 
                                       anno_folder=args.annotations_folder,
                                       transform=transform, 
                                       target=args.target, 
                                       batch_size=args.batch_size, 
                                       num_workers=args.batch_size, 
                                       pin_memory=True,
                                       debug=DEBUG,
                                       debug_size=10)
elif args.dataset == "CFC22" and args.num_prev_frames == 5:
    data_module = NoisyVideoDataModule_prev_frames_5(data_dir=args.base_folder, 
                                                    anno_folder=args.annotations_folder,
                                                    transform=transform, 
                                                    target=args.target, 
                                                    batch_size=args.batch_size, 
                                                    num_workers=args.batch_size, 
                                                    pin_memory=True,
                                                    debug=DEBUG,
                                                    debug_size=10)

if args.model_type == "unet_tiny_3d":
    print(args.dataset)
    if args.dataset == "POCUS":
        data_module = NoisyVideoDataModuleWithWindowSizePOCUS(data_dir=args.base_folder, 
                                                              transform=transform, 
                                                              target=args.target, 
                                                              window_size=args.window_size,
                                                              batch_size=args.batch_size, 
                                                              num_workers=args.batch_size, 
                                                              pin_memory=True,
                                                              debug=DEBUG,
                                                              debug_size=10)
    elif args.dataset == "CVA":
        print("NoisyVideoDataModuleWithWindowSizeCVA")
        data_module = NoisyVideoDataModuleWithWindowSizeCVA(data_dir=args.base_folder, 
                                                              transform=transform, 
                                                              target=args.target, 
                                                              window_size=args.window_size,
                                                              batch_size=args.batch_size, 
                                                              num_workers=args.batch_size, 
                                                              pin_memory=True,
                                                              debug=DEBUG,
                                                              debug_size=10)
    else:
        print("NoisyVideoDataModuleWithWindowSize")
        data_module = NoisyVideoDataModuleWithWindowSize(data_dir=args.base_folder, 
                                                    anno_folder=args.annotations_folder,
                                                    transform=transform, 
                                                    target=args.target, 
                                                    window_size=args.window_size,
                                                    batch_size=args.batch_size, 
                                                    num_workers=args.batch_size, 
                                                    pin_memory=True,
                                                    debug=DEBUG,
                                                    debug_size=10)

# Initialize the model
if args.model_type == "cnn":
    if args.resolution_target == 512 and args.num_prev_frames == 2:
        if not args.with_skip_connections:
            base_model = DenoisingCNN_512(args.fine_layers, args.bottleneck_size)
        else:
            base_model = DenoisingCNN_512_with_skips(args.fine_layers, args.bottleneck_size)
    elif args.resolution_target == 1024 and args.num_prev_frames == 2:
        if not args.with_skip_connections and not args.with_residual_connections:
            base_model = DenoisingCNN_1024(args.fine_layers, args.bottleneck_size)
        elif args.with_skip_connections and not args.with_residual_connections:
            base_model = DenoisingCNN_1024_with_skips(args.fine_layers, args.bottleneck_size)
        if args.with_residual_connections:
            base_model = DenoisingCNN_1024_with_residual_connections(args.fine_layers, args.bottleneck_size)
        if args.with_resnet_blocks:
            base_model = DenoisingCNN_1024_with_skips_and_resnet_blocks(args.fine_layers, args.bottleneck_size)
    elif args.resolution_target == 2048 and args.num_prev_frames == 2:
        if not args.with_skip_connections:
            base_model = DenoisingCNN_2048(args.fine_layers, args.bottleneck_size)
        else:
            base_model = DenoisingCNN_2048_with_skips(args.fine_layers, args.bottleneck_size)
    elif args.resolution_target == 512 and args.num_prev_frames == 5:
        print("DenoisingCNN_512_prev_frames_5")
        base_model = DenoisingCNN_512_prev_frames_5()
    elif args.resolution_target == 1024 and args.num_prev_frames == 5:
        print("DenoisingCNN_1024_prev_frames_5")
        base_model = DenoisingCNN_1024_prev_frames_5()
    else:
        raise ValueError(f"Invalid resolution target: {args.resolution_target} and number of previous frames: {args.num_prev_frames}")
elif args.model_type == "unet":
    print("UNet")
    base_model = UNet()
elif args.model_type == "unet_downscaled":
    print("UNet downscaled")
    base_model = UNet_downscaled()
elif args.model_type == "unet_tiny_3d":
    print("UNet tiny 3d")
    base_model = UNetTiny3D(n_channels=1,
                            n_output_channels=1,
                            n_timesteps_in=args.window_size,
                            num_layers=4,
                            start_hidden_dim=4,
                            bilinear=True
                        )
elif args.model_type == "ae":
    print("AE")
    base_model = AE_1024(in_channels=3)
else:
    raise ValueError(f"Invalid model type: {args.model_type}, only [cnn, unet, resnet50, and vit] are supported")

# setup augmentations
augmentations = Augmentation(
    random_salt_and_pep=args.salt_and_pepper_p,
    random_gauss_blur=args.gaussian_blur_p,
    random_motion_blur=args.motion_blur_p,
    random_erase=args.erase_p,
    random_brightness=args.brightness_p
)

# Initialize the Lightning model
if args.model_type != "unet_tiny_3d":
    if args.num_prev_frames == 2:
        if args.resume:
            model = DenoiserLightning.load_from_checkpoint(os.path.join(args.results_path, 'last_model.ckpt'), 
                                                           model=base_model, 
                                                           inference_clips=[],
                                                           frames_folder=args.base_folder,
                                                           annotations_folder=args.annotations_folder, 
                                                           augmentations=augmentations, 
                                                           experiment_name=os.path.basename(args.results_path), 
                                                           optimizer=args.optimizer, 
                                                           scheduler=args.scheduler, 
                                                           step_size=args.step_size, 
                                                           gamma=args.gamma, 
                                                           factor=args.factor, 
                                                           patience=args.patience)
            # find out what the last model's epoch was
            last_epoch = model.current_epoch
            print(f"Resuming from epoch {last_epoch}")
            args.epochs = args.epochs - last_epoch
        else:
            if args.dataset == "CVA" or args.dataset == "POCUS" or args.dataset == "Fluo":
                inference_clips = []
            else:
                inference_clips = [args.inference_clip1, args.inference_clip2, args.inference_clip3, args.inference_clip4, args.inference_clip5]
            model = DenoiserLightning(
                model=base_model,
                experiment_name=os.path.basename(args.results_path),
                learning_rate=args.learning_rate,
                inference_clips=inference_clips,
                frames_folder=args.base_folder,
                annotations_folder=args.annotations_folder,
                target=args.target,
                loss=args.loss,
                loss_lambda=args.loss_lambda,
                augmentations=augmentations,
                optimizer=args.optimizer,
                scheduler=args.scheduler,
                step_size=args.step_size,
                gamma=args.gamma,
                factor=args.factor,
                patience=args.patience
            )
    elif args.num_prev_frames == 5:
        if args.resume:
            model = DenoiserLightning_prev_frames_5.load_from_checkpoint(os.path.join(args.results_path, 'last_model.ckpt'), model=base_model)
        else:
            model = DenoiserLightning_prev_frames_5(
                model=base_model,
                learning_rate=args.learning_rate,
                inference_clips=[args.inference_clip1, args.inference_clip2, 
                            args.inference_clip3, args.inference_clip4, args.inference_clip5],
                frames_folder=args.base_folder,
                annotations_folder=args.annotations_folder,
                target=args.target
            )
    else:
        raise ValueError(f"Invalid number of previous frames: {args.num_prev_frames}")
# unet_tiny_3d trumps other settings
elif args.model_type == "unet_tiny_3d":
    if args.dataset == "POCUS" or args.dataset == "CVA":
        inference_clips = []
    else: inference_clips = [args.inference_clip1, args.inference_clip2, args.inference_clip3, args.inference_clip4, args.inference_clip5]
    model = DenoiserLightningAlt(
        model=base_model,
        optimizer="adamw",
        learning_rate=args.learning_rate,
        inference_clips=inference_clips,
        frames_folder=args.base_folder,
        annotations_folder=args.annotations_folder,
        target=args.target,
        loss=args.loss,
        loss_lambda=args.loss_lambda,
        augmentations=augmentations
    )

# Logging
run_name = f"{args.dataset}_denoiser_target{args.target}_res{args.resolution_target}_prevfrs{args.num_prev_frames}_bs{args.batch_size}_lr{learning_rate}_epochs{epochs}_input{os.path.basename(args.base_folder)}_fine{args.fine_layers}_bnck{args.bottleneck_size}_adamW_stepLR_{args.model_type}_loss{args.loss}_ll{args.loss_lambda}_opt{args.optimizer}_lr{args.learning_rate}"
if args.with_skip_connections:
    run_name += "_skip"
if args.with_residual_connections:
    run_name += "_residual"
if args.with_resnet_blocks:
    run_name += "_resnet"
if args.scheduler == "step":
    run_name += f"_stepLR_{args.step_size}_{args.gamma}"
elif args.scheduler == "reduce_on_plateau":
    run_name += f"_reduceLR_{args.factor}_{args.patience}"
if not DEBUG and not args.no_logging:
    wandb_logger = WandbLogger(
        project="SAVeD",  
        name=run_name,            
        config={                  # Config to track/save
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "resolution_target": args.resolution_target,
            "num_prev_frames": args.num_prev_frames,
            "base_folder": args.base_folder,
            "model_type": model.__class__.__name__,
            "target": args.target,
            "fine_layers": args.fine_layers,
            "bottleneck_size": args.bottleneck_size,
            "loss": args.loss,
            "loss_lambda": args.loss_lambda
        }
    )
    # args.results_path = f'checkpoints/{run_name}'
    logger = wandb_logger
else:
    logger = None

# Set callbacks to save best and last checkpoints
checkpoint_callback_best = ModelCheckpoint(
    dirpath=f'{args.results_path}',
    filename='best_model',
    save_top_k=1,
    verbose=True,
    monitor='train_loss',
    mode='min'
)
checkpoint_callback_last = ModelCheckpoint(
    dirpath=f'{args.results_path}',
    filename='last_model',
    save_top_k=1,
    verbose=True,
    save_last=True  # Saves the last model
)
lr_monitor = LearningRateMonitor(logging_interval='step')

# set trainer
if DEBUG:
    trainer = pl.Trainer(devices=1, 
                         accelerator='gpu' if torch.cuda.is_available() else 'cpu', 
                         logger=None, 
                         overfit_batches=1, 
                         log_every_n_steps=1, 
                         max_epochs=30, 
                         default_root_dir=args.results_path,
                         num_sanity_val_steps=0)
else:
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=-1,
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=[checkpoint_callback_best, checkpoint_callback_last, lr_monitor],
        logger=logger,
        log_every_n_steps=200,
        default_root_dir=args.results_path
)

# train
trainer.fit(model, data_module)

if not DEBUG and not args.no_logging:
    wandb.finish()