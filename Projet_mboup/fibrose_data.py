import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import os
import sys
path_data_f0 = '/content/IMAGES_ECHO_LABELLISEES/F0/'
path_data_f4 = '/content/IMAGES_ECHO_LABELLISEES/F4/'
path2weights = '/content/drive/MyDrive/models'

sys.path.append(path_data_f0)
sys.path.append(path_data_f4)

path2dataf0 = os.path.join(path_data_f0)
path2dataf4 = os.path.join(path_data_f4) 

full_filenames0 = os.listdir(path2dataf0)
full_filenames4 = os.listdir(path2dataf4)

full_files0 = [os.path.join(path2dataf0,f) for f in full_filenames0]
full_files4 = [os.path.join(path2dataf4,f) for f in full_filenames4]

data=full_files0+full_files4
datadf=pd.DataFrame(columns=["id","path","label","niveau fibrose"])
for i in full_files0:
     i=i.replace(path_data_f0,"")
     datadf=datadf.append({"id":i,"path":path_data_f0,"label":0,"niveau fibrose":"F0"},ignore_index=True)
for i in full_files4:
     i=i.replace(path_data_f4,"")
     datadf=datadf.append({"id":i,"path":path_data_f4,"label":1,"niveau fibrose":"F4"},ignore_index=True)
