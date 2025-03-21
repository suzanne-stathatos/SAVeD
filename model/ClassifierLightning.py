import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import wandb
import os
import pandas as pd
import os
import pickle
import re
from collections import defaultdict

class ClassifierLightning(pl.LightningModule):
    def __init__(self, model, learning_rate=0.0005):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss)
        self.log('train_accuracy', accuracy)
        return {'loss': loss, 'train_accuracy': accuracy}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy)
        return {'val_loss': loss, 'val_accuracy': accuracy}
    
    def on_train_epoch_end(self):
        print("Training epoch end")
        print(f"Training accuracy: {self.trainer.callback_metrics['train_accuracy']}")
        return self.trainer.callback_metrics['train_accuracy']

    def on_validation_epoch_end(self):
        print("Validation epoch end")
        return self.trainer.callback_metrics['val_accuracy']

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
            },
        }