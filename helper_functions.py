import openslide
import cv2
import tensorflow as tf
import numpy as np
import skimage
import os
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from skimage import io

# Normalization png images
def normalize(input_image, real_image):
    """ Normalizing the images to [-1, 1]
    """
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return (input_image), (real_image)

img_width = 256
img_height = 256

def pad_org(image, new_size):
    """
    Padding the image to make the mask and the original has the same position
    """
    if image.shape[0] > image.shape[1]:
        canvas = (tf.ones((image.shape[0], image.shape[0] - image.shape[1], image.shape[-1]), dtype='int32'))*255
        new_img = tf.concat([image, canvas], 1)
    elif image.shape[0] < image.shape[1]:
        canvas = (tf.ones((image.shape[1] - image.shape[0], image.shape[1], image.shape[-1]), dtype='int32'))*255
        new_img = tf.concat([image, canvas], 0)
    else: 
        new_img = image

    result = tf.image.resize(new_img, (new_size, new_size))
    return result 

def pad_mask(image, new_size):
    if image.shape[0] > image.shape[1]:
        canvas = tf.zeros((image.shape[0], image.shape[0] - image.shape[1], image.shape[-1]), dtype='int32')
        new_img = tf.concat([image, canvas], 1)  
        
    elif image.shape[0] < image.shape[1]:
        canvas = tf.zeros((image.shape[1] - image.shape[0], image.shape[1], image.shape[-1]), dtype='int32')
        new_img = tf.concat([image, canvas], 0) 
        
    else:
        new_img = image
    
    result = tf.image.resize(new_img, (new_size, new_size))
    result = tf.cast(result, tf.int32)
    return result 

def pad_pair(original, mask):
    new_inp = pad(original, 'input', 256)
    new_mask = pad(mask, 'mask', 256)

    return new_inp, new_mask

def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, 256, 256, 3])

    return cropped_image[0], cropped_image[1]

@tf.function()
def random_jitter(input_image, real_image):
    """
    Randomly flip the image horizontally
    """  
    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image

def load(image_file):
    """
    Load the PNG image file
    """
    image = tf.io.read_file(image_file)
    image = tf.io.decode_image(image, channels=3)
    image = tf.cast(image, tf.int32)
    return image

def load_tiff(image_file):
    """
    Load the TIFF image file 
    """
    image = openslide.OpenSlide(image_file)
    image = image.get_thumbnail(image.level_dimensions[-1])
    image = np.array(image)
    return image

def load_image(original_file, mask_file):
    """ 
    Loading images from both files 
    """
    original_image = load(original_file)
    mask_image = load(mask_file)
    return original_image, mask_image 

def preprocess_train(original_img, mask_img):
    original_img, mask_img = random_crop(original_img, mask_img)
    original, mask = random_jitter(original_img, mask_img)
    original, mask = normalize(original_img, mask_img)

    return original, mask

def preprocess_test(original_img, mask_img):
    original, mask = random_jitter(original_img, mask_img)
    original, mask = normalize(original_img, mask_img)

    return original, mask


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                                kernel_initializer=initializer, use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


OUTPUT_CHANNELS = 3

def Generator():
    """
    The architecture of generator is a modified U-Net.
    Each block in the encoder is (Conv -> Batchnorm -> Leaky ReLU)
    Each block in the decoder is (Transposed Conv -> Batchnorm -> Dropout(applied to the first 3 blocks) -> ReLU)
    There are skip connections between the encoder and decoder (as in U-Net).
    """
    inputs = tf.keras.layers.Input(shape=[256,256,3])

    down_stack = [
      downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
      downsample(128, 4), # (bs, 64, 64, 128)
      downsample(256, 4), # (bs, 32, 32, 256)
      downsample(512, 4), # (bs, 16, 16, 512)
      downsample(512, 4), # (bs, 8, 8, 512)
      downsample(512, 4), # (bs, 4, 4, 512)
      downsample(512, 4), # (bs, 2, 2, 512)
      downsample(512, 4), # (bs, 1, 1, 512)
    ]

    up_stack = [
      upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
      upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
      upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
      upsample(512, 4), # (bs, 16, 16, 1024)
      upsample(256, 4), # (bs, 32, 32, 512)
      upsample(128, 4), # (bs, 64, 64, 256)
      upsample(64, 4), # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                          strides=2,
                                          padding='same',
                                          kernel_initializer=initializer,
                                          activation='tanh') # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
      x = down(x)
      skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
      x = up(x)
      x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def Discriminator():
    """
    The Discriminator is a PatchGAN.
    Each block in the discriminator is (Conv -> BatchNorm -> Leaky ReLU)
    The shape of the output after the last layer is (batch_size, 30, 30, 1)
    Each 30x30 patch of the output classifies a 70x70 portion of the input image (such an architecture is called a PatchGAN).
    Discriminator receives 2 inputs.
    Input image and the target image, which it should classify as real.
    Input image and the generated image (output of generator), which it should classify as fake.
    We concatenate these 2 inputs together in the code (tf.concat([inp, tar], axis=-1))
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                    kernel_initializer=initializer,
                                    use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                    kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    """
    It is a sigmoid cross entropy loss of the generated images and an array of ones
    This allows the generated image to become structurally similar to the target image.
    """
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    """
    The discriminator loss function takes 2 inputs; real images, generated images
    real_loss is a sigmoid cross entropy loss of the real images and an array of ones(since these are the real images)
    generated_loss is a sigmoid cross entropy loss of the generated images and an array of zeros(since these are the fake images)
    Then the total_loss is the sum of real_loss and the generated_loss
    """
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def generate_images(model, test_input, tar):
    """
    Generating test images for evaluating the result during training
    """
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15,15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')


def get_key(my_dict, val): 
    for key, value in my_dict.items(): 
         if val == value: 
             return key 

def getStages(image_path):
    image = cv2.imread(image_path)
    original = image.copy()
    colors = ['green', 'yellow', 'orange', 'red']
    lower = [[50,100,100],[20,100,100],[18, 40, 90],[161, 155, 84]]
    upper = [[70,255,255],[30, 255, 255],[27, 255, 255],[179, 255, 255]]
    dict_of_area = {}
    for i in range(4):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(image_hsv, np.array(lower[i],dtype="uint8"), np.array(upper[i],dtype="uint8"))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        area = 0
        for c in cnts:
            area += cv2.contourArea(c)
            cv2.drawContours(original,[c], 0, (0,0,0), 2)
        dict_of_area[str(colors[i])] = area
    list_of_area = list(dict_of_area.values())
    list_of_area.sort(reverse=True)
    majority = max(list_of_area)
    major_color = (get_key(dict_of_area, majority))
    if list_of_area[1] != 0:
        minority = list_of_area[1]
    else:
        minority = majority
    minor_color = (get_key(dict_of_area, minority))
    
    stages = ['black', 'gray', 'green', 'yellow', 'orange', 'red']
    majority = stages.index(major_color)
    minority = stages.index(minor_color)
    
    return (majority, minority)

def psnr(x1, x2):
    """
    Calculating the peak-signal-noiser-ratio between a noise-free image and its noisy approximation image 
    """
    return tf.image.psnr(x1, x2, max_val=255)

def evaluate(model, dataset):
    """ 
    Evaluate super-resolution results on PSNR metrics.
    """
    psnr_values = []
    for lr, hr in dataset:
        sr = resolve(model, lr)
        psnr_value = psnr(hr, sr)[0]
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)