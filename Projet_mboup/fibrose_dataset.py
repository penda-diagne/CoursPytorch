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
    img = self.transform(img)
    return img,self.labels[i]
    
