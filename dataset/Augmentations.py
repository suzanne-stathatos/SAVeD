import torch
from torch import nn

import kornia.augmentation as K


class Augmentation(nn.Module):
    def __init__(self, random_salt_and_pep=0.0, random_gauss_blur=0.0, random_motion_blur=0.0, random_erase=0.0, random_brightness=0.0):
        super().__init__()
        # all augmentations initialized with p=0.0
        # save probabilities as attributes
        self.snp_p = random_salt_and_pep
        self.gb_p = random_gauss_blur
        self.mb_p = random_motion_blur
        self.erase_p = random_erase
        self.brightness_p = random_brightness

        # initialize augmentations
        self.random_salt_and_pep = K.RandomSaltAndPepperNoise(amount=0.5, salt_vs_pepper=0.5, p=self.snp_p, same_on_batch=True)
        self.random_gauss_blur = K.RandomGaussianBlur((5, 5), (0.1, 2.0),p=self.gb_p, same_on_batch=True)
        self.random_motion_blur = K.RandomMotionBlur(5, 30., 0.5, p=self.mb_p, same_on_batch=True)
        self.random_erase = K.RandomErasing(scale=(.001, .001), ratio=(.3, 1/.3), p=self.erase_p, same_on_batch=True)
        self.random_brightness = K.RandomBrightness(brightness=(0.2, 2.0), p=self.brightness_p, same_on_batch=True)

        # initialize augmentation pipeline
        self.augmentations = K.AugmentationSequential(
            self.random_salt_and_pep,
            self.random_gauss_blur,
            self.random_motion_blur,
            self.random_erase,
            self.random_brightness,
        )

    def forward(self, frame1, frame2, frame3, target, mask):
        # Check if all augmentations probabilities are 0 -- if so, return original frames
        if self.snp_p == 0.0 and self.gb_p == 0.0 and self.mb_p == 0.0 and self.erase_p == 0.0 and self.brightness_p == 0.0:
            return frame1, frame2, frame3, target, mask
        # otherwise, apply augmentations
        params = self.augmentations.forward_parameters(frame1.shape)

        # Apply same augmentations to all inputs
        frame1_aug = self.augmentations(frame1, params=params)
        frame2_aug = self.augmentations(frame2, params=params)
        frame3_aug = self.augmentations(frame3, params=params)
        target_aug = self.augmentations(target, params=params)
        mask_aug = self.augmentations(mask, params=params)
        
        return frame1_aug, frame2_aug, frame3_aug, target_aug, mask_aug