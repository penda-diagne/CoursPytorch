from argparse import ArgumentParser
import HistoDataModule
import HistoModel

import sys
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

initial_parameters={'input_shape':(3,96,96),"initial_filters":8,"num_fc1":100,"num_classes":2,"dropout_rate":0.25}initial_parameters={'input_shape':(3,96,96),"initial_filters":8,"num_fc1":100,"num_classes":2,"dropout_rate":0.25}
histo_data = HistoDataModule.HistoDataModule(path_data, transform, batch_size=64)
histo_data.setup()
def main(hparams):
    model = HistoModel.model_lit(initial_parameters)
    trainer = Trainer(gpus=hparams.gpus,max_epochs=hparams.max_epochs,num_workers=hparams.num_workers)
    trainer.fit(model,histo_data)
    if __name__ == '__main__':
        parser = ArgumentParser()
        parser.add_argument('--gpus', default=None,required = True)
        parser.add_argument('--max_epochs', default=4,required = True)
        parser.add_argument('--num_workers', default=20,required = True)
        args = parser.parse_args() 
