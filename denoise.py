import torch
# from main import DenoisingCNN
from argparse import ArgumentParser
import glob
import os
from PIL import Image, ImageFile
from skimage import filters
import numpy as np
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm
import cv2

import torch
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from einops import rearrange

from model.DenoisingCNN import DenoisingCNN_512, DenoisingCNN_1024, DenoisingCNN_2048, DenoisingCNN_512_with_skips, DenoisingCNN_1024_with_skips, DenoisingCNN_2048_with_skips, DenoisingCNN_1024_with_residual_connections, DenoisingCNN_1024_with_skips_and_resnet_blocks
from model.UNet import UNet
from model.UNet_downscaled import UNet_downscaled
from model.UNetTiny3D import UNetTiny3D
from model.AE import AE_1024


from dataset.CFCLightlyDataset import CFCLightlyDataset, CFC22_NORMALIZE

from model.DenoisingCNN_prev_frames_5 import DenoisingCNN_512_prev_frames_5, DenoisingCNN_1024_prev_frames_5

ImageFile.LOAD_TRUNCATED_IMAGES = True

def unpatchify ( patches, patch_size, batch_size, channels, height, width ):
    # undo the work done with patchify on the patches to construct images
    height_patches = height // patch_size
    width_patches = width // patch_size
    num_patches = height_patches * width_patches

    patch_dim = channels * patch_size**2
    patches = patches.reshape(shape=(batch_size, height_patches, width_patches, patch_size, patch_size, channels))
    patches = torch.einsum("nhwpqc->nchpwq", patches)
    images = patches.reshape(shape=(batch_size, channels, height, width))
    return images



def load_model(model_path, device, resolution, num_prev_frames, fine_layers, bottleneck_size, model_type, with_skip_connections, with_residual_connections, with_resnet_blocks):
    try:
        # Load the entire checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract the model's state_dict
        state_dict = checkpoint['state_dict']
        
        # Remove the 'model.' prefix and filter out augmentation parameters
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                key = k.replace('model.', '')
                if not key.startswith('augmentations'):  # Skip augmentation parameters
                    filtered_state_dict[key] = v
        
        # Initialize the appropriate model
        if with_residual_connections:
            model = DenoisingCNN_1024_with_residual_connections(fine_layers, bottleneck_size)
        elif with_resnet_blocks:
            model = DenoisingCNN_1024_with_skips_and_resnet_blocks(fine_layers, bottleneck_size)
        elif model_type == 'cnn':
            if resolution == 512 and num_prev_frames == 2:
                if with_skip_connections:
                    model = DenoisingCNN_512_with_skips(fine_layers=fine_layers, bottleneck_size=bottleneck_size)
                else:
                    model = DenoisingCNN_512(fine_layers=fine_layers, bottleneck_size=bottleneck_size)
            elif resolution == 1024 and num_prev_frames == 2:
                if with_skip_connections:
                    model = DenoisingCNN_1024_with_skips(fine_layers=fine_layers, bottleneck_size=bottleneck_size)
                else:
                    model = DenoisingCNN_1024(fine_layers=fine_layers, bottleneck_size=bottleneck_size)
            elif resolution == 2048 and num_prev_frames == 2:
                if with_skip_connections:
                    model = DenoisingCNN_2048_with_skips(fine_layers=fine_layers, bottleneck_size=bottleneck_size)
                else:
                    model = DenoisingCNN_2048(fine_layers=fine_layers, bottleneck_size=bottleneck_size)
            elif resolution == 512 and num_prev_frames == 5:
                model = DenoisingCNN_512_prev_frames_5()
            elif resolution == 1024 and num_prev_frames == 5:
                model = DenoisingCNN_1024_prev_frames_5()
            else:
                raise ValueError(f"Resolution {resolution} and num_prev_frames {num_prev_frames} not supported")
        elif model_type == 'unet':
            model = UNet()
        elif model_type == 'unet_downscaled':
            model = UNet_downscaled()
        elif model_type == 'unet_4hlayers_3d':
            model = UNetTiny3D(n_channels=1,
                            n_output_channels=1,
                            n_timesteps_in=31,
                            num_layers=4,
                            start_hidden_dim=4,
                            bilinear=True
                        )
        elif model_type == 'ae':
            model = AE_1024(in_channels=3)
        else:
            raise ValueError(f"Model type {model_type} not supported")
        
        # Load the filtered state dict
        model.load_state_dict(filtered_state_dict, strict=True)
        model = model.to(device)
        print("Model loaded successfully")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e


def min_max_spatial_normalize(frames):
    '''
    Normalizes each frame to be between 0 and 1 spatially.
    '''
    frames = frames - np.min(frames)
    frames = frames / np.max(frames)

    frs = []
    for fr in frames:
        min_val = np.min(fr)
        max_val = np.max(fr)
        frs.append((fr - min_val) / (max_val - min_val))
    return np.stack(frs)



def denoise(model, input_dir, output_dir, resolution=512, num_prev_frames=2, fine_layers=False, bottleneck_size=16, model_type='cnn', with_skip_connections=False, with_residual_connections=False, with_resnet_blocks=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize((resolution, resolution//2)), transforms.ToTensor()])
    
    # load model from checkpoint
    model = load_model(model, device, resolution, num_prev_frames, fine_layers, 
                       bottleneck_size, model_type, with_skip_connections, with_residual_connections, with_resnet_blocks)
    model.eval()
    
    # Get sorted frames
    print(f"Denoising {input_dir}")
    all_files = sorted(glob.glob(os.path.join(input_dir, '*.jpg')), 
                      key=lambda x: int(x.split("/")[-1].split(".")[0]))
    print(f"Denoising {len(all_files)} frames")
    
    # Process frames sequentially
    all_outputs = []
    with torch.no_grad():
        # save first frame to get image dimensions
        frame1_img = Image.open(all_files[num_prev_frames]).convert('L')
        if model_type == 'unet_4hlayers_3d':
            for i in tqdm(range(31//2, len(all_files) - 31//2)):
                frames = []
                for j in range(31):
                    frames.append(Image.open(all_files[i-31//2+j]).convert('L'))
                frames = [transform(frame).to(device) for frame in frames]
                frames = torch.stack(frames)
                print(f'Denoise frames shape: {frames.shape}')
                frames -= torch.min(frames)
                frames /= torch.max(frames)
                frames = rearrange(frames, 't c h w -> c h w t')
                print(f'Denoise frames shape after rearrange: {frames.shape}')
                # include batch dimension
                frames = frames.unsqueeze(0)
                print(f'Denoise frames shape after unsqueeze: {frames.shape}')
                # make single channel
                # if num_channels == 1:
                frames = frames[:,:1]
                print(f'Denoise frames shape after make single channel: {frames.shape}')
                # Get prediction
                pred = model(frames)
                print(f'Denoise pred shape: {pred.shape}')
                pred = pred.squeeze(-1)
                pred = pred.squeeze(0)
                print(f'Denoise pred shape after squeeze: {pred.shape}')
                output = pred.detach().cpu().numpy().squeeze(0)
                all_outputs.append(output)
        else:
            for i in tqdm(range(num_prev_frames, len(all_files) - num_prev_frames)):
                # Load current and surrounding frames
                frames = []
                frames.append(Image.open(all_files[i]).convert('L'))
                # frame1_img = frames[0]
                
                # Load previous frames
                for j in range(num_prev_frames):
                    frames.append(Image.open(all_files[i - j - 1]).convert('L'))
                # Load future frames    
                for j in range(num_prev_frames):
                    frames.append(Image.open(all_files[i + j + 1]).convert('L'))
                
                # Process frames
                frames = [transform(frame).to(device) for frame in frames]
                # add batch dimension
                frames = [frame.unsqueeze(0) for frame in frames]
                # print(f'Denoise frames shape: {frames[0].shape}')
                x = torch.cat(frames[:num_prev_frames + 1], dim=1)
                # print(f'Denoise x shape: {x.shape}')
                output = model(x)
                # remove batch dimension
                # print(f'Denoise output shape: {output.shape}')
                # Save output
                output = output[0].detach().cpu().numpy().squeeze(0)
                # print(f'Denoise output shape: {output.shape}')
                all_outputs.append(output)
    
    # Normalize and save results
    all_outputs = min_max_spatial_normalize(all_outputs)
    for i, output in enumerate(all_outputs):
        # # Save raw float array without converting to uint8
        # output_path = os.path.join(output_dir, os.path.basename(all_files[i+num_prev_frames]).replace('.jpg', '.npy'))
        # np.save(output_path, output)
        output = (output * 255).astype(np.uint8)
        output = Image.fromarray(output)
        output = output.resize((frame1_img.width, frame1_img.height))
        if model_type == 'ae':
            filename = os.path.basename(all_files[i+num_prev_frames-1])
        else:
            filename = os.path.basename(all_files[i+num_prev_frames])
        output.save(os.path.join(output_dir, filename))


def process_single_clip(clip, args):
    try:
        output_dir = os.path.join(args.output_dir, clip)
        os.makedirs(output_dir, exist_ok=True)
        denoise(args.model, 
               os.path.join(args.input_dir, clip), 
               output_dir, 
               args.resolution_size, 
               args.num_prev_frames,
               args.fine_layers,
               args.bottleneck_size,
               args.model_type,
               args.with_skip_connections, 
               args.with_residual_connections,
               args.with_resnet_blocks)
        return clip, True
    except Exception as e:
        print(f"Error processing clip {clip}: {e}")
        import traceback
        traceback.print_exc()
        return clip, False


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='checkpoints/all_denoised_1024_fine_bnck_512_SKIP/best_model.ckpt')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--resolution_size', type=int, default=512)
    parser.add_argument('--fine_layers', action='store_true')
    parser.add_argument('--bottleneck_size', type=int, default=16)
    parser.add_argument('--num_prev_frames', type=int, default=2)
    parser.add_argument('--model_type', type=str, default='cnn')
    parser.add_argument('--with_skip_connections', action='store_true')
    parser.add_argument('--with_residual_connections', action='store_true')
    parser.add_argument('--with_resnet_blocks', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of clips
    clips = [f for f in os.listdir(args.input_dir) 
             if os.path.isdir(os.path.join(args.input_dir, f))]
    
    # Process clips sequentially
    for clip in tqdm(clips, desc="Processing clips"):
        clip, success = process_single_clip(clip, args)
        if success:
            print(f"Processed {clip}")
        else:
            print(f"Failed {clip}")
    
    # verify that we saved the number of frames we expected
    num_frames = 0
    for clip in clips:
        num_frames += len(glob.glob(os.path.join(args.output_dir, clip, '*.jpg')))
    print(f"Number of frames saved: {num_frames}")
    num_frames_expected = 0
    for clip in clips:
        num_frames_expected += len(glob.glob(os.path.join(args.input_dir, clip, '*.jpg'))) - 2 * args.num_prev_frames
    print(f"Number of frames expected: {num_frames_expected}")
    assert num_frames == num_frames_expected, f"Number of frames saved ({num_frames}) does not match number of frames expected ({num_frames_expected})"
    
    print(f'Denoised output saved to {args.output_dir}')