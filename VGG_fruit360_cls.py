# This file is to create a (multi-class) classifier to identify 
# fruits in the fruit360 dataset. 
# 
# We will use: 
#   A pretrained VGG model; 
#   Transfer learning to train the classifier (following the VGG
# model). 
# Keras in tensorflow 2.0 will be used as the library for deep 
# learning.   
# 
# kliu14, created on 08/10/2020. 

# Do you know what you need to do? Design a framework first, and then
# fill modules in one by one！


# 1. Data： 
# Select three arbitrary fruits, banana, raspberry &strawberry, to 
# demonstrate it is working. 
# The data generator in keras: 
# Expected output (or how to unit test data section): 

# Reference: lecture video on udemy. 
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
# import PIL

from tensorflow.keras.preprocessing.image import load_img

# a. Get the images for visualization
train_path = 'E:\\Coursera\\cv_udemy\\data\\fruits-360\\Training'
test_path = 'E:\\Coursera\\cv_udemy\\data\\fruits-360\\Test'

train_images = glob(train_path + '\*\*.jp*g')
test_images = glob(test_path + '\*\*.jp*g')

print("Size of all the training images:", len(train_images))
print("Size of all the test images:", len(test_images))

sample_img = np.random.choice(train_images)
print(sample_img)
plt.imshow(load_img(sample_img))
plt.show()
sample_img = np.random.choice(test_images)
plt.imshow(load_img(sample_img))
plt.show()
# plt.imshow(load_img(np.random.choice(test_images)))

# b. Get a generator for the model training


# 2. Load the pretrained VGG model
# Reference: https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16. 
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.utils import plot_model

# TODO: specify input dimension!
def create_model():
    from tensorflow.keras.applications import VGG16

    vgg = VGG16(include_top = False, weights='imagenet')
    vgg.summary()

    # 3. Define my model with VGG as a feature transformer
    # VGG16 + dense
    # NOTE: Using functional APIs
    x = layers.Flatten()(vgg.output)
    output = layers.Dense(units = 3, activation='softmax')(x)

    model = keras.Model(inputs = vgg.input, outputs = output)
    return model

# model = create_model()
# plot_model(model)


# 4. Model training


# 5. View loss & accuracy