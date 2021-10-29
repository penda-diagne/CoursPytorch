import torch
from PIL import ImageFilter
from fibrose_data import data
from torch.utils.data import Dataset
import pytorch_lightning as pl
import torchvision.transforms as transforms
import torchmetrics
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
class mboupDataset(Dataset):
  def __init__(self,datadf,data,transform):
    idx = datadf.set_index("id",inplace = True)
    # obtenir les labels du data frame
    self.labels = datadf["label"]

    self.transform = transform

  def __len__(self):
    return len(data)

  def __getitem__(self,i):
    '''
    Ouvre  l'image numero i, applique le transform et retourne avec le label
    '''
    img = Image.open(data[i])
    img_r = img.resize((329,375))
    img_cr = img_r.crop((0, 120, 329, 375))
    img_f = img_cr.filter(ImageFilter.SMOOTH)
    img_t = self.transform(img_f)
    return img_t,self.labels[i]
    
