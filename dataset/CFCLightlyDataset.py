from PIL import Image, ImageOps
import os
import os.path
import torch.utils.data
import numpy as np
import torchvision
import json
import glob
from torchvision.datasets import ImageFolder
from pathlib import Path
import random
from lightly.data import LightlyDataset
from PIL import ImageFile

# accept truncated image files
ImageFile.LOAD_TRUNCATED_IMAGES = True

NUM_CLASSES = 1

CFC22_NORMALIZE = {"mean": [0.200, 0.200, 0.200], "std": [0.189, 0.189, 0.189]}

def default_image_loader(path):
    '''Loads an image from a filepath and returns it as a numpy array'''
    image = Image.open(path).convert('RGB') # convert to RGB to ensure 3 channels
    return image



def generate_metadata_file ( dataset_path ):
    print('Generating metadata file')
    root_path = dataset_path
    if root_path[-1] != '/':
        root_path += '/'

    img_folder = ImageFolder(root_path)
    # load all images from all subdirs of image_folder
    imgs = []


class CFCLightlyDataset(LightlyDataset):
    def __init__ ( self, root_path, custom_transform, debug: bool =False, debug_size: int =1000):
        # if not debug:
        super().__init__(root_path, transform=custom_transform)
        # else:
        #     random.seed(0)
        #     # Recursively get all image files
        #     all_files = []
        #     for ext in ['*.png', '*.jpg', '*.jpeg']:
        #         all_files.extend([
        #                 str(Path(f).relative_to(root_path))
        #                 for f in glob.glob(os.path.join(root_path, '**', ext), recursive=True)
        #             ])

        #     # Randomly sample debug_size images
        #     debug_files = random.sample(all_files, min(debug_size, len(all_files)))
        #     super().__init__(
        #         input_dir=root_path,
        #         transform=custom_transform,
        #         filenames=debug_files
        #     )
        self.root_path = root_path
        self.dataset.loader = default_image_loader