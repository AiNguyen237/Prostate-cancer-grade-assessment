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


# LOAD THE DATASET
# Getting the data directory
train_path = '/home/jupyter/data/train_preprocessed'
masks_path = '/home/jupyter/data/masks_preprocessed'
# Location of the training labels 
train_data = pd.read_csv('/home/jupyter/data/cleaned_ds.csv')
train_data['ori_path'] = train_data['image_id'].map(lambda x: os.path.join(train_path, x+'.png'))
train_data['mask_path'] = train_data['image_id'].map(lambda x: os.path.join(masks_path, x+'.png'))
org_paths = train_data['ori_path'].to_numpy()
mask_paths = train_data['mask_path'].to_numpy()
test_data = train_data.sample(1)
org_sample = test_data['ori_path'].to_numpy()
mask_sample = test_data['mask_path'].to_numpy()

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

# # INPUT PIPELINE 
train_dataset = tf.data.Dataset.from_tensor_slices((org_paths, mask_paths))
train_dataset = train_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.map(preprocess_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices((org_sample, mask_sample))
test_dataset = test_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.map(preprocess_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# BUILD THE GENERATOR
generator = Generator()

# BUILD THE DISCRIMINATOR 
discriminator = Discriminator()

# DEFINE THE OPTIMIZER AND CHECKPOINT-SAVER
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
checkpoint_dir = './training_checkpoints2'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer= generator_optimizer,
                                 discriminator_optimizer= discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 100

log_dir="logs/"

summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)

def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs):
        start = time.time()

        display.clear_output(wait=True)

        # Visualize training process 
        for example_input, example_target in test_dataset.take(1):
            generate_images(generator, example_input, example_target)
            plt.savefig(f'./val_imgs/demo{epoch}.jpeg')
            if epoch == 1:
                psnr_value = evaluate(generator,test_ds)
        print("Epoch: ", epoch)
        
        # Train
        
        for n, (input_image, target) in train_ds.enumerate():
            print('.', end='')
            if (n+1) % 100 == 0:
                print()
            train_step(input_image, target, epoch)
        print()

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            print(f'Ckpt is saved at {epoch}!')

        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
    
    checkpoint.save(file_prefix = checkpoint_prefix)
    print(f'Last ckpt is saved at {epoch}.')

fit(train_dataset, EPOCHS, test_dataset)

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# save model 
final_model = checkpoint.generator
final_model.save('/home/jupyter/model2')
