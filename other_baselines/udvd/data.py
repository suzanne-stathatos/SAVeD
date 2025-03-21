import os
import os.path
import cv2
import glob
import h5py
from PIL import Image
import skimage
import skimage.io

import numpy as np
import pandas as pd

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF

import utils

DATASET_REGISTRY = {}


def build_dataset(name, *args, **kwargs):
    return DATASET_REGISTRY[name](*args, **kwargs)


def register_dataset(name):
    def register_dataset_fn(fn):
        if name in DATASET_REGISTRY:
            raise ValueError("Cannot register duplicate dataset ({})".format(name))
        DATASET_REGISTRY[name] = fn
        return fn

    return register_dataset_fn


@register_dataset("DAVIS")
def load_DAVIS(data, batch_size=100, num_workers=0, image_size=None, stride=64, n_frames=5):
    train_dataset = DAVIS(data, datatype="train", patch_size=image_size, stride=stride, n_frames=n_frames)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)

    valid_dataset = DAVIS(data, datatype="val", n_frames=n_frames)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, num_workers=8, shuffle=False)

    test_dataset = DAVIS(data, datatype="test", n_frames=n_frames)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=8, shuffle=False)
    return train_loader, valid_loader, test_loader

@register_dataset("ImageDAVIS")
def load_ImageDAVIS(data, batch_size=100, num_workers=0, image_size=None, stride=64, n_frames=1):
    train_dataset = ImageDAVIS(data, datatype="train", patch_size=image_size, stride=stride)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    valid_dataset = ImageDAVIS(data, datatype="val")
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, num_workers=4, shuffle=False)

    test_dataset = ImageDAVIS(data, datatype="test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)
    return train_loader, valid_loader, test_loader

@register_dataset("Set8")
def load_Set8(data, batch_size=100, num_workers=0, n_frames=5):
    test_dataset = Set8(data, n_frames=n_frames)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=8, shuffle=False)
    return test_loader

@register_dataset("CTC")
def load_CTC(data, batch_size=100, num_workers=0, image_size=None, stride=64, n_frames=5):
    train_dataset = CTC(data, patch_size=image_size, stride=stride, n_frames=n_frames)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    valid_dataset = CTC(data, n_frames=n_frames)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, num_workers=4, shuffle=False)
    return train_loader, valid_loader

def custom_collate(batch):
    """Custom collate function with size checking"""
    if len(batch) == 0:
        return None
    
    # Ensure all elements are tensors of the same size
    elem = batch[0]
    if not torch.is_tensor(elem):
        raise TypeError(f"Expected tensor, got {type(elem)}")
    
    # Print shapes for debugging
    shapes = [t.shape for t in batch]
    if len(set(shapes)) > 1:
        print(f"Warning: Inconsistent tensor shapes in batch: {shapes}")
        
    try:
        # Stack tensors
        out = torch.stack(batch, dim=0)
        return out
    except RuntimeError as e:
        print(f"Stacking error: {str(e)}")
        # Could add additional handling here if needed
        raise


@register_dataset("CFC")
def load_CFC(data, batch_size=100, num_workers=0, image_size=256, stride=64, n_frames=5):
    train_dataset = CFC(data, patch_size=image_size, stride=stride, n_frames=n_frames)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=custom_collate)

    valid_dataset = CFC(data, n_frames=n_frames)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, num_workers=4, shuffle=False, collate_fn=custom_collate)
    return train_loader, valid_loader

@register_dataset("SingleVideo")
def load_SingleVideo(data, batch_size=8, dataset="DAVIS", video="giant-slalom",image_size=None, stride=64, n_frames=5,
                     aug=0, dist="G", mode="S", noise_std=30, min_noise=0, max_noise=100, sample=False, heldout=False):

    train_dataset = SingleVideo(data, dataset=dataset, video=video, patch_size=image_size, stride=stride, n_frames=n_frames,
                            aug=aug, dist=dist, mode=mode, noise_std=noise_std, min_noise=min_noise, max_noise=max_noise,
                            sample=sample, heldout=heldout
                           )
    test_dataset = SingleVideo(data, dataset=dataset, video=video, patch_size=None, stride=stride, n_frames=n_frames,
                               aug=0, dist=dist, mode=mode, noise_std=noise_std, min_noise=min_noise, max_noise=max_noise,
                               sample=False, heldout=False
                              )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False)
    return train_loader, test_loader

@register_dataset("Nanoparticles")
def load_Nanoparticles(data, batch_size=8, image_size=None, stride=64, n_frames=5, aug=0):

    train_dataset = Nanoparticles(data, datatype="train", patch_size=image_size, stride=stride, n_frames=n_frames, aug=aug)
    test_dataset = Nanoparticles(data, datatype="test", patch_size=None, stride=200, n_frames=n_frames, aug=0)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False)
    return train_loader, test_loader

@register_dataset("RawVideo")
def load_RawVideo(data, batch_size=8, image_size=None, stride=64, n_frames=5, aug=0, scenes=[7, 8, 9, 10, 11], isos = [1600, 3200, 6400, 12800, 25600]):

    train_dataset = RawVideo(data, datatype="train", patch_size=image_size, stride=stride, n_frames=n_frames, aug=aug, scenes=scenes, isos=isos)
    valid_dataset = RawVideo(data, datatype="val", patch_size=1080, stride=1920-1080, n_frames=n_frames, aug=0, scenes=scenes, isos=isos)
    test_dataset = RawVideo(data, datatype="test", patch_size=None, stride=64, n_frames=n_frames, aug=0, scenes=scenes, isos=isos)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, num_workers=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False)
    return train_loader, valid_loader, test_loader


class DAVIS(torch.utils.data.Dataset):
    def __init__(self, data_path, datatype="train", patch_size=None, stride=64, n_frames=5):
        super().__init__()
        self.data_path = data_path
        self.datatype = datatype
        self.size = patch_size
        self.stride = stride
        self.n_frames = n_frames

        if self.datatype == "train":
            self.folders = pd.read_csv(os.path.join(data_path, "ImageSets", "2017", "train.txt"), header=None)
        elif self.datatype == "val":
            self.folders = pd.read_csv(os.path.join(data_path, "ImageSets", "2017", "val.txt"), header=None)
        else:
            self.folders = pd.read_csv(os.path.join(data_path, "ImageSets", "2017", "test-dev.txt"), header=None)
        self.len = 0
        self.bounds = []

        for folder in self.folders.values:
            files = sorted(glob.glob(os.path.join(data_path, "JPEGImages", "480p", folder[0], "*.jpg")))
            self.len += len(files)
            self.bounds.append(self.len)

        if self.size is not None:
            self.n_H = (int((480-self.size)/self.stride)+1)
            self.n_W = (int((854-self.size)/self.stride)+1)
            self.n_patches = self.n_H * self.n_W
            self.len *= self.n_patches

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.size is not None:
            patch = index % self.n_patches
            index = index // self.n_patches

        ends = 0
        x = (self.n_frames-1) // 2
        for i, bound in enumerate(self.bounds):
            if index < bound:
                folder = self.folders.values[i][0]
                if i>0:
                    index -= self.bounds[i-1]
                    newbound = bound - self.bounds[i-1]
                else:
                    newbound = bound
                if(index < x):
                    ends = x-index
                elif(newbound-1-index < x):
                    ends = -(x-(newbound-1-index))
                break

        files = sorted(glob.glob(os.path.join(self.data_path, "JPEGImages", "480p", folder, "*.jpg")))

        Img = Image.open(files[index])
        Img = np.array(Img)

        for i in range(1,x+1):
            end = max(0, ends)
            off = max(0,i-x+end)
            img = Image.open(files[index-i+off])
            img = np.array(img)
            Img = np.concatenate((img, Img), axis=2)

        for i in range(1,x+1):
            end = -min(0,ends)
            off = max(0,i-x+end)
            img = Image.open(files[index+i-off])
            img = np.array(img)
            Img = np.concatenate((Img, img), axis=2)

        if self.size is not None:
            nh = (patch // self.n_W)*self.stride
            nw = (patch % self.n_W)*self.stride
            Img = Img[nh:(nh+self.size), nw:(nw+self.size), :]

        return self.transform(np.array(Img)).type(torch.FloatTensor)

class ImageDAVIS(torch.utils.data.Dataset):
    def __init__(self, data_path, datatype="train", patch_size=None, stride=40):
        super().__init__()
        self.data_path = data_path
        self.datatype = datatype
        self.size = patch_size
        self.stride = stride

        if self.datatype == "train":
            self.folders = pd.read_csv(os.path.join(data_path, "ImageSets", "2017", "train.txt"), header=None)
        elif self.datatype == "val":
            self.folders = pd.read_csv(os.path.join(data_path, "ImageSets", "2017", "val.txt"), header=None)
        else:
            self.folders = pd.read_csv(os.path.join(data_path, "ImageSets", "2017", "test-dev.txt"), header=None)
        self.len = 0
        self.bounds = []

        for folder in self.folders.values:
            files = sorted(glob.glob(os.path.join(data_path, "JPEGImages", "480p", folder[0], "*.jpg")))
            self.len += len(files)
            self.bounds.append(self.len)

        if self.size is not None:
            self.n_H = (int((480-self.size)/self.stride)+1)
            self.n_W = (int((854-self.size)/self.stride)+1)
            self.n_patches = self.n_H * self.n_W
            self.len *= self.n_patches

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.size is not None:
            patch = index % self.n_patches
            index = index // self.n_patches

        for i, bound in enumerate(self.bounds):
            if index < bound:
                folder = self.folders.values[i][0]
                if i>0:
                    index -= self.bounds[i-1]
                break

        files = sorted(glob.glob(os.path.join(self.data_path, "JPEGImages", "480p", folder, "*.jpg")))

        Img = np.array(Image.open(files[index]))

        if self.size is not None:
            nh = (patch // self.n_W)*self.stride
            nw = (patch % self.n_W)*self.stride
            Img = Img[nh:(nh+self.size), nw:(nw+self.size), :]

        return self.transform(Img).type(torch.FloatTensor)
    
class Set8(torch.utils.data.Dataset):
    def __init__(self, data_path, n_frames=5, hop=1):
        super().__init__()
        self.data_path = data_path
        self.len = 0
        self.bounds = []
        self.hop = hop
        self.n_frames = n_frames

        self.folders = []
        self.folders += sorted(glob.glob(os.path.join(data_path, "GoPro/snowboard")))
        self.folders += sorted(glob.glob(os.path.join(data_path, "GoPro/hypersmooth")))
        self.folders += sorted(glob.glob(os.path.join(data_path, "GoPro/rafting")))
        self.folders += sorted(glob.glob(os.path.join(data_path, "GoPro/motorbike")))
        self.folders += sorted(glob.glob(os.path.join(data_path, "Derfs/tractor")))
        self.folders += sorted(glob.glob(os.path.join(data_path, "Derfs/sunflower")))
        self.folders += sorted(glob.glob(os.path.join(data_path, "Derfs/touchdown")))
        self.folders += sorted(glob.glob(os.path.join(data_path, "Derfs/park_joy")))
        for folder in self.folders:
            files = sorted(glob.glob(os.path.join(folder, "*.png")))
            self.len += len(files)
            self.bounds.append(self.len)

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        ends = 0
        x = ((self.n_frames-1) // 2)*self.hop
        for i, bound in enumerate(self.bounds):
            if index < bound:
                folder = self.folders[i]
                if i>0:
                    index -= self.bounds[i-1]
                    newbound = bound - self.bounds[i-1]
                else:
                    newbound = bound
                if(index < x):
                    ends = x-index
                elif(newbound-1-index < x):
                    ends = -(x-(newbound-1-index))
                break
    
        files = sorted(glob.glob(os.path.join(folder, "*.png")))
        
        Img = Image.open(files[index])
        Img = np.array(Img)

        for i in range(self.hop, x+1, self.hop):
            end = max(0, ends)
            off = max(0,i-x+end)
            img = Image.open(files[index-i+off])
            img = np.array(img)
            Img = np.concatenate((img, Img), axis=2)
        
        for i in range(self.hop, x+1, self.hop):
            end = -min(0,ends)
            off = max(0,i-x+end)
            img = Image.open(files[index+i-off])
            img = np.array(img)
            Img = np.concatenate((Img, img), axis=2)

        return self.transform(Img).type(torch.FloatTensor)

class CTC(torch.utils.data.Dataset):
    def __init__(self, data_path, patch_size=None, stride=64, n_frames=5):
        super().__init__()
        self.data_path = data_path
        self.size = patch_size
        self.stride = stride
        self.len = 0
        self.bounds = [0]
        self.nHs = []
        self.nWs = []
        self.n_frames = n_frames

        parent_folders = sorted([x for x in glob.glob(os.path.join(data_path, "*/*")) if os.path.isdir(x)])
        self.folders = []
        for folder in parent_folders:
            self.folders.append(os.path.join(folder, "01"))
            self.folders.append(os.path.join(folder, "02"))
        for folder in self.folders:
            files = sorted(glob.glob(os.path.join(folder, "*.tif")))
            if self.size is not None:
                (h, w) = np.array(cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)).shape
                nH = (int((h-self.size)/self.stride)+1)
                nW = (int((w-self.size)/self.stride)+1)
                self.len += len(files)*nH*nW
                self.nHs.append(nH)
                self.nWs.append(nW)
            else:
                self.len += len(files)
            self.bounds.append(self.len)

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        ends = 0
        x = (self.n_frames-1) // 2
        for i, bound in enumerate(self.bounds):
            if index < bound:
                folder = self.folders[i-1]
                index -= self.bounds[i-1]
                newbound = bound - self.bounds[i-1]
                if self.size is not None:
                    nH = self.nHs[i-1]
                    nW = self.nWs[i-1]
                    patch = index % (nH*nW)
                    index = index // (nH*nW)
                    newbound = newbound // (nH*nW)
                if(index < x):
                    ends = x-index
                elif(newbound-1-index < x):
                    ends = -(x-(newbound-1-index))
                break

        files = sorted(glob.glob(os.path.join(folder, "*.tif")))

        img = cv2.imread(files[index], cv2.IMREAD_GRAYSCALE)
        (h, w) = np.array(img).shape
        Img = np.reshape(np.array(img), (h,w,1))

        for i in range(1,x+1):
            end = max(0, ends)
            off = max(0,i-x+end)
            img = cv2.imread(files[index-i+off], cv2.IMREAD_GRAYSCALE)
            img = np.reshape(np.array(img), (h,w,1))
            Img = np.concatenate((img, Img), axis=2)

        for i in range(1,x+1):
            end = -min(0,ends)
            off = max(0,i-x+end)
            img = cv2.imread(files[index+i-off], cv2.IMREAD_GRAYSCALE)
            img = np.reshape(np.array(img), (h,w,1))
            Img = np.concatenate((Img, img), axis=2)

        if self.size is not None:
            nh = (patch // nW)*self.stride
            nw = (patch % nW)*self.stride
            Img = Img[nh:(nh+self.size), nw:(nw+self.size), :]

        return self.transform(Img).type(torch.FloatTensor)


def numerical_sort(filename):
    """Extract the number from the filename for proper numerical sorting"""
    return int(os.path.splitext(os.path.basename(filename))[0])


class CFC(torch.utils.data.Dataset):
    def __init__(self, data_path, patch_size=None, stride=64, n_frames=5):
        super().__init__()
        self.data_path = data_path
        self.size = patch_size
        self.stride = stride
        self.len = 0
        self.bounds = [0]
        self.nHs = []
        self.nWs = []
        self.n_frames = n_frames

        # get all clip directories
        self.folders = sorted([x for x in glob.glob(os.path.join(data_path, "*")) if os.path.isdir(x)])

        for folder in self.folders:
            # get all frames in the clip
            # IF FISH:
            files = sorted(glob.glob(os.path.join(folder, "*.jpg")), key=numerical_sort)
            # IF POTUS:
            # files = sorted(glob.glob(os.path.join(folder, "*.png")))

            assert len(files) > 0, f"No frames found in {folder}"

            if self.size is not None:
                # read first frame to get shape
                (h, w) = np.array(cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)).shape
                nH = (int((h-self.size)/self.stride)+1)
                nW = (int((w-self.size)/self.stride)+1)
                self.len += len(files)*nH*nW
                self.nHs.append(nH)
                self.nWs.append(nW)
            else:
                self.len += len(files)
            self.bounds.append(self.len)

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        ends = 0
        x = (self.n_frames-1) // 2
        for i, bound in enumerate(self.bounds):
            if index < bound:
                folder = self.folders[i-1]
                index -= self.bounds[i-1]
                newbound = bound - self.bounds[i-1]
                if self.size is not None:
                    nH = self.nHs[i-1]
                    nW = self.nWs[i-1]
                    patch = index % (nH*nW)
                    index = index // (nH*nW)
                    newbound = newbound // (nH*nW)
                if(index < x):
                    ends = x-index
                elif(newbound-1-index < x):
                    ends = -(x-(newbound-1-index))
                break
        # IF FISH:
        files = sorted(glob.glob(os.path.join(folder, "*.jpg")), key=numerical_sort)
        # IF POTUS:
        # files = sorted(glob.glob(os.path.join(folder, "*.png")))

        center_filename = files[index]

        # Load center frame
        img = cv2.imread(files[index], cv2.IMREAD_GRAYSCALE)
        (h, w) = np.array(img).shape
        
        # Initialize padding variables
        pad_bottom, pad_right = 0, 0
        nh, nw = 0, 0
        
        # Calculate patch position and padding if needed
        if self.size is not None:
            nh = (patch // nW)*self.stride
            nw = (patch % nW)*self.stride
            # Add padding if needed
            pad_bottom = max(0, (nh + self.size) - h)
            pad_right = max(0, (nw + self.size) - w)
            if pad_bottom > 0 or pad_right > 0:
                img = np.pad(img, ((0, pad_bottom), (0, pad_right)), mode='reflect')
                h, w = img.shape

        Img = np.reshape(np.array(img), (h,w,1))

        # Load and concatenate previous frames
        for i in range(1,x+1):
            end = max(0, ends)
            off = max(0,i-x+end)
            img = cv2.imread(files[index-i+off], cv2.IMREAD_GRAYSCALE)
            if pad_bottom > 0 or pad_right > 0:
                img = np.pad(img, ((0, pad_bottom), (0, pad_right)), mode='reflect')
            img = np.reshape(np.array(img), (h,w,1))
            Img = np.concatenate((img, Img), axis=2)

        # Load and concatenate next frames
        for i in range(1,x+1):
            end = -min(0,ends)
            off = max(0,i-x+end)
            img = cv2.imread(files[index+i-off], cv2.IMREAD_GRAYSCALE)
            if pad_bottom > 0 or pad_right > 0:
                img = np.pad(img, ((0, pad_bottom), (0, pad_right)), mode='reflect')
            img = np.reshape(np.array(img), (h,w,1))
            Img = np.concatenate((Img, img), axis=2)

        # Extract patch if size is specified
        if self.size is not None:
            Img = Img[nh:(nh+self.size), nw:(nw+self.size), :]
            
            # Verify patch size
            if Img.shape[:2] != (self.size, self.size):
                raise RuntimeError(f"Invalid patch size: {Img.shape[:2]}, expected: ({self.size}, {self.size})")

        final_tensor = self.transform(Img).type(torch.FloatTensor)
        return final_tensor, center_filename # comment in for inference



class SingleVideo(torch.utils.data.Dataset):
    def __init__(self, data_path, dataset="DAVIS", video="giant-slalom", patch_size=None, stride=64, n_frames=5,
                 aug=0, dist="G", mode="S", noise_std=30, min_noise=0, max_noise=100, sample=True, heldout=False):
        super().__init__()
        self.data_path = data_path
        self.dataset = dataset
        self.size = patch_size
        self.stride = stride
        self.n_frames = n_frames
        self.aug = aug
        self.heldout = heldout

        if dataset == "DAVIS":
            self.files = sorted(glob.glob(os.path.join(data_path, "JPEGImages", "480p", video, "*.jpg")))
        elif dataset == "GoPro" or dataset == "Derfs":
            self.files = sorted(glob.glob(os.path.join(data_path, video, "*.png")))
        elif dataset == "Vid3oC":
            self.files = sorted(glob.glob(os.path.join(data_path, "TrainingHR", video, "*.png")))
        elif dataset == "Nanoparticles":
            self.files = sorted(glob.glob(os.path.join(data_path, "*.png")))
            self.noisy_files = sorted(glob.glob(os.path.join(data_path, "*.npy")))

        self.len = self.bound = len(self.files)
        if self.heldout:
            self.len -= 5
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.reverse = transforms.Compose([transforms.ToPILImage()])

        Img = Image.open(self.files[0])
        Img = np.array(Img)
        if dataset == "Nanoparticles":
            H, W = Img.shape
        else:
            H, W, C = Img.shape

        if not dataset == "Nanoparticles":
            os.makedirs(os.path.join(data_path, f"Noisy_Videos_{int(noise_std)}"), exist_ok=True)
            os.makedirs(os.path.join(data_path, f"Noisy_Videos_{int(noise_std)}", video), exist_ok=True)

            self.noisy_folder = os.path.join(data_path, f"Noisy_Videos_{int(noise_std)}", video)

            if sample:
                for i in range(self.len):
                    Img = Image.open(self.files[i])
                    Img = self.transform(Img)
                    self.C, self.H, self.W = Img.shape
                    Noise = utils.get_noise(Img, dist=dist, mode=mode, min_noise=min_noise, max_noise=max_noise, noise_std=noise_std).numpy()
                    Img = Img + Noise
                    np.save(os.path.join(self.noisy_folder, os.path.basename(self.files[i])[:-3]+".npy"), Img)
            self.noisy_files = sorted(glob.glob(os.path.join(self.noisy_folder, "*.npy")))

        if self.size is not None:
            self.n_H = (int((H-self.size)/self.stride)+1)
            self.n_W = (int((W-self.size)/self.stride)+1)
            self.n_patches = self.n_H * self.n_W
            self.len *= self.n_patches

        self.hflip = transforms.Compose([transforms.RandomHorizontalFlip(p=1)])
        self.vflip = transforms.Compose([transforms.RandomVerticalFlip(p=1)])

        if aug >= 1: # Horizonatal and Vertical Flips
            self.len *= 4
        if aug >= 2: # Reverse the Video
            self.len *= 2
        if aug >= 3: # Variable Frame Rate
            self.len *= 4

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        hop = 1
        reverse = 0
        flip = 0
        if self.aug >= 3: # Variable Frame Rate
            hop = index % 4 + 1
            index = index // 4
        if self.aug >= 2: # Reverse the Video
            reverse = index % 2
            index = index // 2
        if self.aug >= 1: # Horizonatal and Vertical Flips
            flip = index % 4
            index = index // 4

        if self.size is not None:
            patch = index % self.n_patches
            index = index // self.n_patches

        ends = 0
        x = ((self.n_frames-1) // 2)*hop
        if index < x:
            ends = x - index
        elif self.bound-1-index < x:
            ends = -(x-(self.bound-1-index))

        Img = Image.open(self.files[index])
        Img = np.array(Img)
        if self.dataset == "Nanoparticles":
            H, W = Img.shape
        else:
            H, W, C = Img.shape
        if self.dataset == "Nanoparticles":
            Img = Img.reshape(H, W, 1)
        noisy_Img = np.load(self.noisy_files[index])

        for i in range(hop, x+1, hop):
            end = max(0, ends)
            off = max(0,i-x+end)
            img = Image.open(self.files[index-i+off])
            img = np.array(img)
            if self.dataset == "Nanoparticles":
                img = img.reshape(H, W, 1)
            noisy_img = np.load(self.noisy_files[index-i+off])
            if reverse == 0:
                Img = np.concatenate((img, Img), axis=2)
                noisy_Img = np.concatenate((noisy_img, noisy_Img), axis=0)
            else:
                Img = np.concatenate((Img, img), axis=2)
                noisy_Img = np.concatenate((noisy_Img, noisy_img), axis=0)

        for i in range(hop, x+1, hop):
            end = -min(0,ends)
            off = max(0,i-x+end)
            img = Image.open(self.files[index+i-off])
            img = np.array(img)
            if self.dataset == "Nanoparticles":
                img = img.reshape(H, W, 1)
            noisy_img = np.load(self.noisy_files[index+i-off])
            if reverse == 0:
                Img = np.concatenate((Img, img), axis=2)
                noisy_Img = np.concatenate((noisy_Img, noisy_img), axis=0)
            else:
                Img = np.concatenate((img, Img), axis=2)
                noisy_Img = np.concatenate((noisy_img, noisy_Img), axis=0)

        if self.size is not None:
            nh = (patch // self.n_W)*self.stride
            nw = (patch % self.n_W)*self.stride
            Img = Img[nh:(nh+self.size), nw:(nw+self.size), :]
            noisy_Img = noisy_Img[:, nh:(nh+self.size), nw:(nw+self.size)]

        if flip == 1:
            Img = np.flip(Img, 1)
            noisy_Img = np.flip(noisy_Img, 2)
        elif flip == 2:
            Img = np.flip(Img, 0)
            noisy_Img = np.flip(noisy_Img, 1)
        elif flip == 3:
            Img = np.flip(Img, (1,0))
            noisy_Img = np.flip(noisy_Img, (2,1))

        return self.transform(np.array(Img)).type(torch.FloatTensor), torch.from_numpy(noisy_Img.copy())

class Nanoparticles(torch.utils.data.Dataset):
    def __init__(self, data_path, datatype="train", patch_size=None, stride=64, n_frames=5, aug=0):
        super().__init__()
        self.data_path = data_path
        self.size = patch_size
        self.stride = stride
        self.n_frames = n_frames
        self.datatype = datatype
        self.aug = aug

        self.files = sorted(glob.glob(os.path.join(data_path, "*.npy")))
        if datatype == "train":
            self.files = self.files[0:35]
        elif datatype == "test":
            self.files = self.files

        self.len = self.bound = len(self.files)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.reverse = transforms.Compose([transforms.ToPILImage()])

        Img = np.load(self.files[0])
        C, H, W = Img.shape

        if self.size is not None:
            self.n_H = (int((H-self.size)/self.stride)+1)
            self.n_W = (int((W-self.size)/self.stride)+1)
            self.n_patches = self.n_H * self.n_W
            self.len *= self.n_patches

        self.hflip = transforms.Compose([transforms.RandomHorizontalFlip(p=1)])
        self.vflip = transforms.Compose([transforms.RandomVerticalFlip(p=1)])

        if aug >= 1: # Horizonatal and Vertical Flips
            self.len *= 4
        if aug >= 2: # Reverse the Video
            self.len *= 2
        if aug >= 3: # Variable Frame Rate
            self.len *= 4

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        hop = 1
        reverse = 0
        flip = 0
        if self.aug >= 3: # Variable Frame Rate
            hop = index % 4 + 1
            index = index // 4
        if self.aug >= 2: # Reverse the Video
            reverse = index % 2
            index = index // 2
        if self.aug >= 1: # Horizonatal and Vertical Flips
            flip = index % 4
            index = index // 4

        if self.size is not None:
            patch = index % self.n_patches
            index = index // self.n_patches

        ends = 0
        x = ((self.n_frames-1) // 2)*hop
        if index < x:
            ends = x - index

        Img = np.load(self.files[index])
        C, H, W = Img.shape

        for i in range(hop, x+1, hop):
            end = max(0, ends)
            off = max(0,i-x+end)
            img = np.load(self.files[index-i+off])
            if reverse == 0:
                Img = np.concatenate((img, Img), axis=0)
            else:
                Img = np.concatenate((Img, img), axis=0)
                
        if self.bound-1-index < x:
            ends = -(x-(self.bound-1-index))

        for i in range(hop, x+1, hop):
            end = -min(0,ends)
            off = max(0,i-x+end)
            img = np.load(self.files[index+i-off])
            if reverse == 0:
                Img = np.concatenate((Img, img), axis=0)
            else:
                Img = np.concatenate((img, Img), axis=0)

        if self.size is not None:
            nh = (patch // self.n_W)*self.stride
            nw = (patch % self.n_W)*self.stride
            Img = Img[:, nh:(nh+self.size), nw:(nw+self.size)]

        if flip == 1:
            Img = np.flip(Img, 2)
        elif flip == 2:
            Img = np.flip(Img, 1)
        elif flip == 3:
            Img = np.flip(Img, (2,1))

        return torch.from_numpy(Img.copy()).type(torch.FloatTensor)
    
class RawVideo(torch.utils.data.Dataset):
    def __init__(self, data_path, datatype="train", patch_size=None, stride=64, n_frames=5, aug=0,
                 scenes=[7, 8, 9, 10, 11],
                 isos = [1600, 3200, 6400, 12800, 25600]):
        super().__init__()
        self.data_path = data_path
        self.datatype = datatype
        self.size = patch_size
        self.stride = stride
        self.n_frames = n_frames
        self.aug = aug
        
        self.noisy_path = os.path.join(self.data_path, "indoor_raw_noisy")
        self.gt_path = os.path.join(self.data_path, "indoor_raw_gt")
        self.scenes = scenes
        self.isos = isos
        if self.datatype == "train":
            self.nr = 9 # noise_realisations
        elif self.datatype == "val":
            self.nr = 1 # only the 9th noise realisation used for heldout
        elif self.datatype == "test":
            self.nr = 10
        self.fpv = self.bound = 7 # frames_per_video
        
        self.len = self.fpv * self.nr * len(self.isos) * len(self.scenes)
        
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.reverse = transforms.Compose([transforms.ToPILImage()])

        Img = skimage.io.imread(os.path.join(self.noisy_path, f"scene{self.scenes[0]}", 
                                             f"ISO{self.isos[0]}", "frame1_noisy0.tiff"))
        H, W = Img.shape
        
        if self.size is not None:
            self.n_H = (int((H-self.size)/self.stride)+1)
            self.n_W = (int((W-self.size)/self.stride)+1)
            self.n_patches = self.n_H * self.n_W
            self.len *= self.n_patches

        self.hflip = transforms.Compose([transforms.RandomHorizontalFlip(p=1)])
        self.vflip = transforms.Compose([transforms.RandomVerticalFlip(p=1)])

        if aug >= 1: # Horizonatal and Vertical Flips
            self.len *= 4
        if aug >= 2: # Reverse the Video
            self.len *= 2
        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        hop = 1
        reverse = 0
        flip = 0
        if self.aug >= 2: # Reverse the Video
            reverse = index % 2
            index = index // 2
        if self.aug >= 1: # Horizonatal and Vertical Flips
            flip = index % 4
            index = index // 4

        if self.size is not None:
            patch = index % self.n_patches
            index = index // self.n_patches
            
        scene = index % len(self.scenes)
        index = index // len(self.scenes)
        
        iso = index % len(self.isos)
        index = index // len(self.isos)
        
        if self.datatype == "val":
            nr = 9
        else:
            nr = index % self.nr
            index = index // self.nr
        
        ends = 0
        x = ((self.n_frames-1) // 2)*hop
        if index < x:
            ends = x - index
        elif self.bound-1-index < x:
            ends = -(x-(self.bound-1-index))

        Img = skimage.io.imread(os.path.join(self.gt_path, 
                                             f"scene{self.scenes[scene]}",
                                             f"ISO{self.isos[iso]}", 
                                             f"frame{index+1}_clean_and_slightly_denoised.tiff"))
        H, W = Img.shape
        Img = Img.reshape(H, W, 1)
        noisy_Img = skimage.io.imread(os.path.join(self.noisy_path, 
                                             f"scene{self.scenes[scene]}",
                                             f"ISO{self.isos[iso]}", 
                                             f"frame{index+1}_noisy{nr}.tiff"))
        noisy_Img = noisy_Img.reshape(H, W, 1)

        for i in range(hop, x+1, hop):
            end = max(0, ends)
            off = max(0,i-x+end)
            # img = Image.open(self.files[index-i+off])
            img = skimage.io.imread(os.path.join(self.gt_path, 
                                             f"scene{self.scenes[scene]}",
                                             f"ISO{self.isos[iso]}", 
                                             f"frame{index-i+off+1}_clean_and_slightly_denoised.tiff"))
            img = img.reshape(H, W, 1)
            # noisy_img = np.load(self.noisy_files[index-i+off])
            noisy_img = skimage.io.imread(os.path.join(self.noisy_path, 
                                             f"scene{self.scenes[scene]}",
                                             f"ISO{self.isos[iso]}", 
                                             f"frame{index-i+off+1}_noisy{nr}.tiff"))
            noisy_img = noisy_img.reshape(H, W, 1)
            
            if reverse == 0:
                Img = np.concatenate((img, Img), axis=2)
                noisy_Img = np.concatenate((noisy_img, noisy_Img), axis=2)
            else:
                Img = np.concatenate((Img, img), axis=2)
                noisy_Img = np.concatenate((noisy_Img, noisy_img), axis=2)

        for i in range(hop, x+1, hop):
            end = -min(0,ends)
            off = max(0,i-x+end)
            # img = Image.open(self.files[index+i-off])
            img = skimage.io.imread(os.path.join(self.gt_path, 
                                             f"scene{self.scenes[scene]}",
                                             f"ISO{self.isos[iso]}", 
                                             f"frame{index+i-off+1}_clean_and_slightly_denoised.tiff"))
            img = img.reshape(H, W, 1)
            # noisy_img = np.load(self.noisy_files[index+i-off])
            noisy_img = skimage.io.imread(os.path.join(self.noisy_path, 
                                             f"scene{self.scenes[scene]}",
                                             f"ISO{self.isos[iso]}", 
                                             f"frame{index+i-off+1}_noisy{nr}.tiff"))
            noisy_img = noisy_img.reshape(H, W, 1)
            
            if reverse == 0:
                Img = np.concatenate((Img, img), axis=2)
                noisy_Img = np.concatenate((noisy_Img, noisy_img), axis=2)
            else:
                Img = np.concatenate((img, Img), axis=2)
                noisy_Img = np.concatenate((noisy_img, noisy_Img), axis=2)

        if self.size is not None:
            nh = (patch // self.n_W)*self.stride
            nw = (patch % self.n_W)*self.stride
            Img = Img[nh:(nh+self.size), nw:(nw+self.size), :]
            noisy_Img = noisy_Img[nh:(nh+self.size), nw:(nw+self.size), :]

        if flip == 1:
            Img = np.flip(Img, 1)
            noisy_Img = np.flip(noisy_Img, 1)
        elif flip == 2:
            Img = np.flip(Img, 0)
            noisy_Img = np.flip(noisy_Img, 0)
        elif flip == 3:
            Img = np.flip(Img, (1,0))
            noisy_Img = np.flip(noisy_Img, (1,0))
            
        Img = Img.astype(np.float32)
        noisy_Img = noisy_Img.astype(np.float32)
        Img = (Img-240)/(2**12-1-240)
        noisy_Img = (noisy_Img-240)/(2**12-1-240)

        return self.transform(np.array(Img)).type(torch.FloatTensor), self.transform(np.array(noisy_Img)).type(torch.FloatTensor)
