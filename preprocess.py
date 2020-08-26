from helper_functions import *
import numpy as np
# LOAD THE DATASET
# Getting the data directory
train_path = '/home/jupyter/data/originals'
masks_path = '/home/jupyter/data/new_masks'
# Location of the training labels 
train_data = pd.read_csv('/home/jupyter/data/cleaned_ds.csv')
train_data['ori_path'] = train_data['image_id'].map(lambda x: os.path.join(train_path, x+'.png'))
train_data['mask_path'] = train_data['image_id'].map(lambda x: os.path.join(masks_path, x+'.png'))

org_paths = train_data['ori_path'].to_numpy()
mask_paths = train_data['mask_path'].to_numpy()

for path in mask_paths:
    images_id = path.split('/')[-1]
    img = load(path)
    new_img = pad_mask(img, 500) 
    path_2 = os.path.join('./data/masks_resized', images_id) 
    tf.keras.preprocessing.image.save_img(path_2,new_img)

for path in org_paths:
    images_id = path.split('/')[-1]
    img = load(path)
    new_img = pad_org(img, 500) 
    path_2 = os.path.join('./data/train_resized', images_id) 
    tf.keras.preprocessing.image.save_img(path_2,new_img)