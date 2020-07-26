# import tensorflow as tf
from tensorflow import keras
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical


print()
print("Hello, TF2 world!")
print()

# Load the Dataset: 
def load_data():
    """
    """
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print("type(train_images) =", type(train_images))
    print("train_images.shape =", train_images.shape)
    print("type(train_labels) =", type(train_labels))
    print("len(train_labels) =", len(train_labels))
    print("train_labels[0] =", train_labels[0])

    return train_images, train_labels, test_images, test_labels

# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images

    return train_norm, test_norm

# Define the Model:
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))

    # Compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Model training and evaluateion:
#   Split the data into training / testing (or K-fold split) 
#   Evaluate its accuracy on the training & testing dataset. 
# Get a training curve for the model performance. 
# 
# If we apply K-fold: 
# For each split k (K-splits in all), we get a (Xtrain_k, ytrain_k), (Xvalid_k, yvalid_k):
#   for each epoch i: 
#       train the model using (Xtrain_k, ytrain_k), 
#       valid the model using (Xvalid_k, yvalid_k),
#       record the acc_train & acc_valid.
#   get a history = [acc_train_list, acc_valid_list]

from sklearn.model_selection import KFold
def evaluate_model(dataX, dataY, n_folds = 5):
    scores, histories = [], []
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    for train_ix, valid_ix in kfold.split(dataX):
        trainX, trainY, validX, validY = dataX[train_ix], dataY[train_ix], dataX[valid_ix], dataY[valid_ix]

        # NOTE: fit() take either numpy array, tf tensors, or tf Dataset. 
        # Here numpy arrays are provided. 
        model = create_model()
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, 
                            validation_data=(validX, validY),
                            verbose=0)
        _, acc = model.evaluate(validX, validY, verbose=0)
        print('> %.3f' % (acc*100))

        scores.append(acc)
        histories.append(history)

    return scores, history

train_images, train_labels, test_images, test_labels = load_data()
train_images, test_images = prep_pixels(train_images, test_images)


import time
t1 = time.clock()

print()
print("Evaluate Model:")
scores, histories = evaluate_model(train_images, train_labels)
print("len(scores) =", len(scores))
print("scores[0] =", scores[0], "scores[-1] =", scores[-1])
print("histories[-1] =", histories[-1])

print()
print("Total execution time:", time.clock() - t1)