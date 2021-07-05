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
