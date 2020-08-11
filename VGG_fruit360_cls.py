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
# Expected output: 
#  

# 2. Load the pretrained VGG model
# Reference: https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16. 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import plot_model

# TODO: specify input dimension!
vgg = VGG16(include_top = False, weights='imagenet')
vgg.summary()

# 3. Define my model with VGG as a feature transformer
# VGG16 + dense
# NOTE: Using functional APIs
x = layers.Flatten()(vgg.output)
output = layers.Dense(units = 3, activation='softmax')(x)

model = keras.Model(inputs = vgg.input, outputs = output)
plot_model(model)



# 4. Model training

# 5. View loss & accuracy