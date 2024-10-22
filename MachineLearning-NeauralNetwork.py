#Eugene Yakubu
#Machine Learning Project
# Importing necessary libraries for data processing, image manipulation, and machine learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import itertools
import cv2 as cv
import os

# Configure numpy to display arrays with a precision of 7 decimal places
np.set_printoptions(precision=7)

# TensorFlow and Keras imports for deep learning tasks
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

# Sklearn utilities for handling data splits, confusion matrices, and labels
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.datasets import cifar10

# Keras Model and layer definitions
from keras import Model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator

# Transfer learning models: VGG19 and ResNet50
from keras.applications import VGG19, ResNet50 
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Flatten, Dense, BatchNormalization, Activation, Dropout
from keras.utils import to_categorical
from keras import Sequential

# Additional utilities for model visualization and PIL for image processing
from collections import Counter
from keras.utils import to_categorical, plot_model
from keras import backend as K
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from PIL import Image

# Operating system utilities for file handling and error management
import os
import errno

# Try to create a directory to store datasets
try:
    data_dir = 'dataset'
    os.mkdir(data_dir)
except OSError as e:
    if e.errno == errno.EEXIST:
        print('Directory already created.')
    else:
        raise

# Define the dataset name
dataset_name = "cifar10"

# Load CIFAR-10 dataset from TensorFlow Datasets (TFDS)
(train_set, test_set), dataset_info = tfds.load(
    name=dataset_name, 
    split=["train", "test"], 
    with_info=True, 
    data_dir=data_dir
)

# Print dataset info, which provides metadata about the dataset
print(dataset_info)

# Extract class names from the dataset info
class_names = dataset_info.features["label"].names

# Display the shape, data type, and number of classes in the dataset
print('Image shape    :', dataset_info.features['image'].shape)
print('Image dtype    :', dataset_info.features['image'].dtype)
print()
print('Number of classes:', dataset_info.features["label"].num_classes)
print('Class labels    :', dataset_info.features["label"].names)
print()
print('Number of training samples:', dataset_info.splits["train"].num_examples)
print('Number of testing samples  :', dataset_info.splits["test"].num_examples)

# Display an example from the training set
fig = tfds.show_examples(train_set, dataset_info)

# Define the input shape for the model (resized to 80x80 pixels, 3 color channels)
input_shape = (80, 80, 3)

# Prepare training data by resizing images and converting them to numpy arrays
X_train = []
y_train = []

for example in tfds.as_numpy(train_set):
    new_img = example['image']
    new_img = cv.resize(new_img, input_shape[:2], interpolation=cv.INTER_AREA) 
    X_train.append(new_img)
    y_train.append(example['label'])

# Free up memory by deleting the original training set
del train_set

# Convert the lists to numpy arrays
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

# Print the shape of the training data
print('X_train.shape =', X_train.shape)
print('y_train.shape =', y_train.shape)

# Prepare testing data by resizing images and converting them to numpy arrays
X_test = []
y_test = []

for example in tfds.as_numpy(test_set):
    new_img = example['image']
    new_img = cv.resize(new_img, input_shape[:2], interpolation=cv.INTER_AREA)
    X_test.append(new_img)
    y_test.append(example['label'])

# Free up memory by deleting the original testing set
del test_set

# Convert the lists to numpy arrays
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

# Print the shape of the test data
print('X_test.shape =', X_test.shape)
print('y_test.shape =', y_test.shape)

# Split off a validation set from the training data (using the last 300 samples)
X_val = X_train[-300:]
y_val = y_train[-300:]

# Use the rest of the data as the new training set
X_train = X_train[:-300]
y_train = y_train[:-300]

# Print the shapes of the new training, validation, and test sets
print('X_train.shape =', X_train.shape)
print('y_train.shape =', y_train.shape)
print('\nX_val.shape  =', X_val.shape)
print('y_val.shape  =', y_val.shape)
print('\nX_test.shape  =', X_test.shape)
print('y_test.shape  =', y_test.shape)

# Convert labels to one-hot encoding for training, validation, and test sets
y_train_hot = to_categorical(y_train, 102)
y_val_hot = to_categorical(y_val, 102)
y_test_hot = to_categorical(y_test, 102)

# Print the shapes of the one-hot encoded labels
print('y_train_hot.shape =', y_train_hot.shape)
print('y_val_hot.shape   =', y_val_hot.shape)
print('y_test_hot.shape  =', y_test_hot.shape)

# Load the InceptionV3 model pre-trained on ImageNet without the top layers (for transfer learning)
in_model = InceptionV3(
    include_top=False, 
    weights="imagenet", 
    input_shape=(80, 80, 3)
)

# Add new layers on top of InceptionV3 for classification
x = in_model.layers[-1].output
x = GlobalAveragePooling2D()(x)  # Global pooling to reduce feature size
predictions = Dense(102, activation='softmax')(x)  # Output layer with softmax for classification

# Define the final model
in_model = Model(inputs=in_model.input, outputs=predictions)

# Print the model summary
in_model.summary()

# Plot the model architecture
plot_model(in_model, show_shapes=True, show_layer_names=False, rankdir='LR', dpi=20)

# Compile the model with Adam optimizer and categorical crossentropy loss
in_model.compile(
    loss='categorical_crossentropy', 
    optimizer=tf.keras.optimizers.Adam(), 
    metrics=['accuracy']
)

# Define data augmentation strategies for the training data
datagen = ImageDataGenerator(
    rotation_range=15,  # Rotate images by up to 15 degrees
    width_shift_range=0.1,  # Shift images horizontally by 10%
    height_shift_range=0.1,  # Shift images vertically by 10%
    shear_range=0.1,  # Apply shearing transformations
    zoom_range=0.1,  # Zoom in/out by 10%
    channel_shift_range=0.1,  # Randomly shift the color channels
    horizontal_flip=True  # Randomly flip images horizontally
)

# Define a learning rate schedule function
def lr_schedule(epoch):
    lr = 1e-3  # Start with a learning rate of 0.001
    if epoch > 30:
        lr *= 0.01  # Reduce learning rate after 30 epochs
    elif epoch > 20:
        lr *= 0.1  # Reduce learning rate after 20 epochs
    return lr

# Define a learning rate callback to adjust the learning rate during training
lr_callback = LearningRateScheduler(lr_schedule)

# Define a model checkpoint callback to save the model with the best validation accuracy
myCheckpoint = ModelCheckpoint(filepath='./dataset/my_model.h5', monitor='val_accuracy', save_best_only=True)

# Define batch size and number of epochs
batch_size = 64
epochs = 12

# Create an augmented training set
augmented_train = datagen.flow(X_train, y_train_hot, batch_size)

# Train the model using the augmented data
history = in_model.fit(
    augmented_train,
    validation_data=(X_val, y_val_hot),
    epochs=epochs, 
    steps_per_epoch=len(X_train) // 64,
    callbacks=[lr_callback, myCheckpoint],
    verbose=2
)

# Plot training and validation accuracy over epochs
plt.rcParams['figure.figsize'] = [7, 5]
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate the model on the test set
scores = in_model.evaluate(X_test, y_test_hot)
print('Test loss    :', scores[0])
print('Test accuracy: %.2f%%' % (scores[1] * 100))

# Further evaluation on train, validation, and test sets
train_scores = in_model.evaluate(X_train, y_train_hot)
test_scores = in_model.evaluate(X_test, y_test_hot)
val_scores = in_model.evaluate(X_val, y_val_hot)

# Load pre-trained weights into the model from a saved file (here, './dataset/my_model.h5')
in_model.load_weights('./dataset/my_model.h5')

# Evaluate the model's performance on the test dataset (X_test, y_test_hot), 
# which returns the test loss and test accuracy as a list (scores).
scores = in_model.evaluate(X_test, y_test_hot)

# Print the test loss and test accuracy. The accuracy is multiplied by 100 to display as a percentage.
print('Test loss    :', scores[0])
print('Test accuracy: %.2f%%' % (scores[1] * 100))

# Evaluate the model's performance on the training dataset (X_train, y_train_hot)
train_scores = in_model.evaluate(X_train, y_train_hot)

# Evaluate the model's performance on the test dataset (X_test, y_test_hot)
test_scores = in_model.evaluate(X_test, y_test_hot)

# Evaluate the model's performance on the validation dataset (X_val, y_val_hot)
val_scores = in_model.evaluate(X_val, y_val_hot)

# Print the training loss and accuracy. The accuracy is multiplied by 100 to display as a percentage.
print('Train Loss: %.5f with Accuracy: %.1f%%' % (train_scores[0], (train_scores[1] * 100)))

# Print the test loss and accuracy. The accuracy is multiplied by 100 to display as a percentage.
print('Test  Loss: %.5f with Accuracy: %.1f%%' % (test_scores[0], (test_scores[1] * 100)))

# Print the validation loss and accuracy. The accuracy is multiplied by 100 to display as a percentage.
print('Val   Loss: %.5f with Accuracy: %.1f%%' % (val_scores[0], (val_scores[1] * 100)))

# Download test images from various URLs and save them locally as 'data_test_0.jpg' to 'data_test_4.jpg'
!wget -O 'data_test_0.jpg' 'https://scx1.b-cdn.net/csz/news/800/2018/2-dog.jpg'
!wget -O 'data_test_1.jpg' 'https://static.toiimg.com/thumb/msid-67586673,width-800,height-600,resizemode-75,imgsize-3918697,pt-32,y_pad-40/67586673.jpg'
!wget -O 'data_test_2.jpg' 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/Green_tree_frog.jpg/799px-Green_tree_frog.jpg'
!wget -O 'data_test_3.jpg' 'https://bsmedia.business-standard.com/_media/bs/img/article/2019-10/15/full/1571086349-8577.jpg'
!wget -O 'data_test_4.jpg' 'https://www.om.org/img/h55955_42-62.jpg'

# Download and load the CIFAR-10 dataset (x_train, Y_train for training; x_test, Y_test for testing)
(x_train, Y_train), (x_test, Y_test) = cifar10.load_data()

# Split the training dataset into training (70%) and validation (30%) sets
x_train, x_val, Y_train, Y_val = train_test_split(x_train, Y_train, test_size=0.3)

# Print the shapes of the training, validation, and test datasets before one-hot encoding
print((x_train.shape, Y_train.shape))
print((x_val.shape, Y_val.shape))
print((x_test.shape, Y_test.shape))

# Perform one-hot encoding of the labels, converting them from integers to a one-hot encoded format
# This is necessary because the model outputs 10 classes, and the labels need to match that dimension
Y_train = to_categorical(Y_train)
Y_val = to_categorical(Y_val)
Y_test = to_categorical(Y_test)

# Verify the shape of the datasets after one-hot encoding
# We expect the label arrays (Y_train, Y_val, Y_test) to have their second dimension change to 10
print((x_train.shape, Y_train.shape))
print((x_val.shape, Y_val.shape))
print((x_test.shape, Y_test.shape))

# Image Data Augmentation for the training, validation, and test datasets.
# This helps the model generalize better by applying random transformations to the training images.
train_generator = ImageDataGenerator(
                                    rotation_range=2,  # Randomly rotate the images by 2 degrees
                                    horizontal_flip=True,  # Randomly flip images horizontally
                                    zoom_range=.1)  # Randomly zoom in by 10%

val_generator = ImageDataGenerator(
                                    rotation_range=2, 
                                    horizontal_flip=True,
                                    zoom_range=.1)

test_generator = ImageDataGenerator(
                                    rotation_range=2, 
                                    horizontal_flip=True,
                                    zoom_range=.1) 

# Apply the augmentation settings to the training, validation, and test datasets
train_generator.fit(x_train)
val_generator.fit(x_val)
test_generator.fit(x_test)

# Define a learning rate annealer, which reduces the learning rate when the validation accuracy stops improving
lrr = ReduceLROnPlateau(
                        monitor='val_acc',  # The metric to be monitored
                        factor=.01,  # Factor by which the learning rate will be reduced
                        patience=3,  # Number of epochs with no improvement before reducing the learning rate
                        min_lr=1e-5)  # The minimum learning rate to which it can be reduced

# Define the VGG19 convolutional neural network model (without the top dense layers),
# pre-trained on the ImageNet dataset and with input images of size (32, 32, 3)
base_model = VGG19(include_top=False, weights='imagenet', input_shape=(32, 32, 3), classes=Y_train.shape[1])

# Add final classification layers to the VGG19 base model
vgg_model = Sequential()
vgg_model.add(base_model)  # Add the pre-trained VGG19 model
vgg_model.add(Flatten())  # Flatten the output to feed it into dense (fully connected) layers

# Print the summary of the model architecture so far
vgg_model.summary()

# Add fully connected (Dense) layers with batch normalization and ReLU activation
vgg_model.add(Dense(1024, activation='relu', input_dim=512))
vgg_model.add(Dense(512, activation='relu'))
vgg_model.add(Dense(256, activation='relu'))
vgg_model.add(Dense(128, activation='relu'))

# Add the final output layer with 10 units (one for each class) and softmax activation for classification
vgg_model.add(Dense(10, activation='softmax'))

# Print the final summary of the complete model
vgg_model.summary()

# Define training parameters
vgg_batch_size = 100
epochs = 20
learn_rate = 0.001

# Define the optimizers, starting with Stochastic Gradient Descent (SGD)
sgd = SGD(lr=learn_rate, momentum=0.9, nesterov=False)

# Optionally, you can use the Adam optimizer instead
adam = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# Compile the model with the specified loss function, optimizer, and evaluation metrics
# The loss function is categorical crossentropy since the labels are one-hot encoded
vgg_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using the training and validation datasets
# The 'train_generator' and 'val_generator' flow augmented data in batches to the model
vgg_model.fit(train_generator.flow(x_train, Y_train, batch_size=100),
              epochs=epochs,
              steps_per_epoch=x_train.shape[0] // vgg_batch_size,
              validation_data=val_generator.flow(x_val, Y_val, batch_size=100),
              validation_steps=250,
              callbacks=[lrr], verbose=1)

# Loop through the 5 downloaded test images and make predictions
for i in range(5):
    # Open and display each image
    new_img = Image.open('data_test_' + str(i) + '.jpg')
    new_img = np.array(new_img)
    new_img2 = cv.resize(new_img, input_shape[:2], interpolation=cv.INTER_AREA)
    plt.imshow(new_img2)
    plt.axis('off')
    plt.show()

    # Preprocess the image and make predictions using the trained model
    new_img2 = np.expand_dims(new_img2, 0).astype(np.float64)
    pred = model.predict(new_img2)

    # Get the predicted class ID and print the predicted class
    class_id = np.argmax(pred)
    print('Predicted ID:', class_id)
    print('Class Prediction:', class_names[class_id])

