import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset.CVADataset import CVADataset
from typing import Optional

class CVADataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
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
        self.transform = transform
        self.target = target
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.debug = debug
        self.debug_size = debug_size
         

    def setup(self, stage: Optional[str] = None):
        """Load dataset."""
        if stage == "fit" or stage is None:
            full_dataset = CVADataset(
                base_folder=self.data_dir,
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
            pin_memory=self.pin_memory
        )