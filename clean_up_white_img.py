from helper_functions import *
import numpy as np
import tensorflow as tf
import cv2

# LOAD THE DATASET
# Getting the data directory
train_path = '/home/jupyter/data/originals'
masks_path = '/home/jupyter/data/masks'
# Location of the training labels 
train_data = pd.read_csv('/home/jupyter/data/new_train.csv')
radboud_data = train_data[train_data['data_provider'] == 'radboud']
radboud_data['ori_path'] = train_data['image_id'].map(lambda x: os.path.join(train_path, x+'.png'))
radboud_data['mask_path'] = train_data['image_id'].map(lambda x: os.path.join(masks_path, x+'.png'))
org_paths = radboud_data['ori_path'].to_numpy()

list_of_noise = []

for i in org_paths:
    img = cv2.imread(i)
    img = cv2.resize(img, (256, 256))
    if np.sum(img) == 0 or np.sum(img) == 255:
        list_of_noise.append(i)
    else:
        continue
        
cleaned_ds = radboud_data[~radboud_data.ori_path.isin(list_of_noise)]
cleaned_ds.to_csv('/home/jupyter/data/cleaned_ds.csv')


