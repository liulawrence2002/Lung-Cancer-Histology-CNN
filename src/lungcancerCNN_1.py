# this is code to use CNN to classify histology slides based on if they are 1: adenocarcinoma. 2: benighn 3: squamous cell carcinoma. 

import os 
import imageio
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense , Flatten 




image_folders = {'adenocarcinoma':1 , 'benign':2 , 'squamous_cell_carcinoma' :3 }

image_data =[]

for folder_name , label in image_folders.items():
    folder_path = os.path.join('images/', folder_name)
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.jpg'):
            image_data.append({'image_name': file_name , 'label':label})

df = pd.DataFrame(image_data)

df.to_excel('image_label_index.xlsx' , index = False)

