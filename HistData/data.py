import sys
import torch
from torch.utils.data import Dataset
import os
import torchvision.transforms as transforms
path_data_train = '/content/train'
path_data_test = '/content/test'
path_data = '/content/'
path2weights = '/content/models'
path_helper = '/content/drive/MyDrive/Colab Notebooks/BinaryClassifcation/histo'
sys.path.append(path_data_train)
sys.path.append(path_data_test)
sys.path.append(path_data)
sys.path.append(path_helper)
sys.path.append(path2weights)

import pandas as pd
import matplotlib.pyplot as plt
import HistoDataModule
labels = pd.read_csv(path_data+'train_labels.csv')


class HistoDataset(Dataset):
  def __init__(self,path_data,transform,data_type ="train"):
    
    #Obtenir le path des données d'image
    path2data = os.path.join(path_data,data_type)
    # Obtenir une liste de données
    full_filenames = os.listdir(path2data)
    # Obtenir les chemins vers les images
    self.full_files = [os.path.join(path2data,f) for f in full_filenames]
    # Les labels sont dans un fichier csv denommé train_labels.csv
    
    label_frame = pd.read_csv(path_data+'train_labels.csv')
    # Mettre l'index du data frame à id
    idx = label_frame.set_index("id",inplace = True)
    # obtenir les labels du data frame
    self.labels = [label_frame.loc[filename[:-4]]['label'] for filename in full_filenames]

    self.transform = transform
   


  def __len__(self):
    return len(self.full_files)

  def __getitem__(self,i):
    '''
    Ouvre  l'image numero i, applique le transform et retourne avec le label
    '''
    img = Image.open(self.full_files[i])
    img = self.transform(img)
    return img,self.labels[i]
  
transform = transforms.Compose([
     
    transforms.ToTensor(),
])

thedata = HistoDataset(path_data,transform,"train")
histo_data = HistoDataModule.DataModule(path_data, transform, batch_size=64)
histo_data.setup()
