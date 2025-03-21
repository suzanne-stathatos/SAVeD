import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset.NoisyVideoDataset_prev_frames_5 import NoisyVideoDataset_prev_frames_5
from typing import Optional

class NoisyVideoDataModule_prev_frames_5(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        anno_folder: str,
        transform=None,
        target: str = "l1",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        debug: bool = False,
        debug_size: int = 1000
    ):
        super().__init__()
        self.data_dir = data_dir
        self.anno_folder = anno_folder
        self.transform = transform
        self.target = target
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.debug = debug
        self.debug_size = debug_size
        self.save_hyperparameters(ignore=['transform'])

    def setup(self, stage: Optional[str] = None):
        """Load dataset."""
        if stage == "fit" or stage is None:
            full_dataset = NoisyVideoDataset_prev_frames_5(
                base_folder=self.data_dir,
                anno_folder=self.anno_folder,
                transform=self.transform,
                target=self.target
            )
            if self.debug:
                # Take a small subset of the data
                from torch.utils.data import Subset
                import random
                
                indices = list(range(len(full_dataset)))
                random.shuffle(indices)
                indices = indices[:self.debug_size]
                self.train_dataset = Subset(full_dataset, indices)
                print(f"Debug mode: Using {self.debug_size} samples")
            else:
                self.train_dataset = full_dataset
            print(f"Training size: {len(self.train_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )