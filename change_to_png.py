from helper_functions import load_tiff
import os
import cv2
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio
from helper_functions import visualize_whole
import matplotlib.pyplot as plt
import skimage
import matplotlib
import openslide
import numpy as np

# LOADING THE DATASET 
# Getting the data directory
train_path = '/home/jupyter/data/train/train_images'
masks_path = '/home/jupyter/data/train/train_label_masks'
# Location of the training labels 
train_data = pd.read_csv('/home/jupyter/data/cleaned_ds.csv')
for _, r in train_data.iterrows():
    img = r['image_id']
    masks = os.path.join(masks_path, img+'_mask.tiff')
    cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])
    image = load_tiff(masks)
    path = os.path.join('./data/new_masks', img+'.png')
    plt.imsave(path, np.asarray(image)[:,:,0], cmap=cmap, vmin=0, vmax=5)

for _, r in train_data.iterrows():
    img = r['image_id']
    originals = os.path.join(train_path, img+'.tiff')
    image = load_tiff(originals)
    path = os.path.join('./data/originals', img+'.png')
    plt.imsave(path, np.asarray(image))
