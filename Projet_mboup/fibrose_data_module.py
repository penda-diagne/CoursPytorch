import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
import torchvision.transforms as transforms
import torchmetrics.functionnal.accuracy as accuracy
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.models as models
from torch.nn import functional as F
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import os
class Caltech101DataModule(pl.LightningDataModule):
    def __init__(self, batch_size,dataset):
        super().__init__()
        self.batch_size = batch_size
        self.dataset=dataset
        # Augmentation policy
        self.augmentation = transforms.Compose([
              transforms.RandomHorizontalFlip(),
              transforms.Resize(size=(329, 375)),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        self.transform = transforms.Compose([
              transforms.Resize(size=(329, 375)),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        
        self.num_classes = 2

    def setup(self,stage=None):
        len_data=len(self.dataset)
        len_train=int(0.6*len_data)
        len_val=int(0.2*len_data)
        len_test=len_data-len_train-len_val
        self.train, self.val, self.test = random_split(self.dataset,[len_train,len_val,len_test])
        self.train.dataset.transform = self.augmentation
        self.val.dataset.transform = self.transform
        self.test.dataset.transform = self.transform
        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
