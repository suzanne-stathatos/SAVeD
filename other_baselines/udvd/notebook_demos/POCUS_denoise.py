import os
import os.path
import cv2
import glob
import h5py
import tqdm
import argparse
import logging
from PIL import Image 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
sns.set_theme()
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

import sys
sys.path.append('../')
import data, utils, models

parallel = True
Fast = False
pretrained = True
old = True
load_opt = False

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
transform = transforms.Compose([transforms.ToPILImage()])
to_gray = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(num_output_channels=1)])
center = transforms.Compose([transforms.CenterCrop(40)])

dataset = "CFC"
patch_size = 256
stride = 128
is_image = False
n_frames = 5
cpf = 1
mid = n_frames // 2
is_real = True

aug = 3

dist = 'G'
mode = 'S'
noise_std = 30
min_noise = 0
max_noise = 100

batch_size = 1
lr = 1e-4



def denoise(model_path, input_dir, output_dir):    
    model, optimizer, args = utils.load_model(model_path, parallel=parallel, pretrained=pretrained, old=old, load_opt=load_opt)
    model.to(device)
    model.eval()
    print(f'output dir: {output_dir}')

    # create a test_loader for the clip
    # BEFORE RUNNING THIS, MODIFY THE data.build_dataset() FUNCTION TO RETURN THE FILENAMES
    train_loader, test_loader = data.build_dataset("CFC", input_dir, batch_size=1, image_size=None)

    with torch.no_grad():
        print(len(test_loader.dataset))
        # iterate over full test_loader dataset
        for i in tqdm(range(len(test_loader))):
            noisy_inputs, noisy_filename = test_loader.dataset[i]
            # get directory of noisy_filename to know clip_name
            original_framesize = cv2.imread(noisy_filename, cv2.IMREAD_GRAYSCALE).shape
            # get clip_name and filename
            clip_name = os.path.basename(os.path.dirname(noisy_filename))
            print(f'clip_name: {clip_name}')
            filename = os.path.basename(noisy_filename)
            print(f'filename: {filename}')
            file_to_save = os.path.join(output_dir, clip_name, filename)
            os.makedirs(os.path.dirname(file_to_save), exist_ok=True)
            print(f'file_to_save: {file_to_save}')
            # convert noisy_input to device
            noisy_inputs = noisy_inputs.to(device)
            # add batch dimension
            noisy_inputs_with_batch = noisy_inputs.unsqueeze(0)
            # get outputs
            outputs, est_sigma = model(noisy_inputs_with_batch)
            # get noisy frame
            # noisy_frame = noisy_inputs_with_batch[:, (mid * cpf):((mid + 1) * cpf), :, :]
            # get outputs
            np_outputs = outputs[0,0,:,:].cpu().detach().numpy()
            # scale outputs to 0-1
            np_outputs = np_outputs - np.min(np_outputs)
            np_outputs = np_outputs / np.max(np_outputs)
            # resize to original framesize
            np_outputs = cv2.resize(np_outputs, (original_framesize[1], original_framesize[0]))
            # save the noisy frame as Image scaled from 0 to 255
            np_outputs = (np_outputs * 255).astype(np.uint8)
            noisy_frame = Image.fromarray(np_outputs)
            noisy_frame.save(file_to_save, quality=95)
            print(f'saved {file_to_save}')
            assert os.path.exists(file_to_save), f"File {file_to_save} does not exist after it was saved"



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/root/Data/CFC22/frames/raw/kenai-val')
    parser.add_argument('--output_dir', type=str, default='/root/Data/CFC22/udvd_frames_denoised/kenai-val')
    parser.add_argument('--model', type=str, default='/root/SAVeD/checkpoints/udvd/checkpoint_best.pt')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of clips
    clips = [f for f in os.listdir(args.input_dir) 
             if os.path.isdir(os.path.join(args.input_dir, f))]
    print(f'output dir: {args.output_dir}')
    denoise(args.model, args.input_dir, args.output_dir)

    # verify that we saved the number of frames we expected
    num_frames = 0
    for clip in clips:
        num_frames += len(glob.glob(os.path.join(args.output_dir, clip, '*.png')))
    print(f"Number of frames saved: {num_frames}")
    num_frames_expected = 0
    for clip in clips:
        num_frames_expected += len(glob.glob(os.path.join(args.input_dir, clip, '*.png')))
    print(f"Number of frames expected: {num_frames_expected}")
    assert num_frames == num_frames_expected, f"Number of frames saved ({num_frames}) does not match number of frames expected ({num_frames_expected})"
    
    print(f'Denoised output saved to {args.output_dir}')
