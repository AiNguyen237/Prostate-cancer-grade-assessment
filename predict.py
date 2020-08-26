# IMPORT LIBRARIES 
import tensorflow as tf
import pandas as pd 
import numpy as np

import os
import time
import datetime

from matplotlib import pyplot as plt
from IPython import display
from helper_functions import *

test_dataset = tf.data.Dataset.from_tensor_slices((org_sample, mask_sample))
test_dataset = test_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.map(preprocess_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.shuffle(BUFFER_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model_2 = tf.keras.models.load_model('model2')

i = 1
while i < 10:
    for example_input, example_target in test_dataset.take(1):
        prediction = model_2(example_input, training=True)
        plt.figure(figsize=(15,15))
        plt.imshow(prediction[0] * 0.5 + 0.5)
        plt.axis('off')
        plt.savefig(f'./result/result{i}.jpeg')
        i += 1