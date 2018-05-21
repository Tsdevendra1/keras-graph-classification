import tensorflow as tf
import keras
import os
import re
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from PIL import Image

class ClassProperty (property):
    """Subclass property to make classmethod properties possible"""
    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()

class Model:

    def __init__(self, training_path, validation_path, batch_size, num_epochs):
        self.training_path = training_path
        self.validation_path = validation_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_training_examples = None
        self.num_validation_examples = None

    def define_folder_sizes(self):
        positives_folder_training, negatives_folder_training = os.listdir(self.training_path)
        positives_folder_validation, negatives_folder_validation = os.listdir(self.validation_path)

        num_training_examples = len(os.listdir(os.path.join(self.training_path, positives_folder_training))) + \
                                len(os.listdir(os.path.join(self.training_path, negatives_folder_training)))

        num_validation_examples = len(os.listdir(os.path.join(self.validation_path, positives_folder_validation))) + \
                                len(os.listdir(os.path.join(self.validation_path, negatives_folder_validation)))

        self.num_training_examples = num_training_examples
        self.num_validation_examples = num_validation_examples

    # @staticmethod
    # @property
    @ClassProperty
    @classmethod
    def prediction(cls):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

        model.add(Conv2D(32, (3, 3), dim_ordering='tf'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), dim_ordering='tf'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])
        return model

    @property
    def optimise(self):

        self.define_folder_sizes()

        model = self.prediction

        # this is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        # this is the augmentation configuration we will use for testing:
        # only rescaling
        test_datagen = ImageDataGenerator(rescale=1. / 255)

         # this is a generator that will read pictures found in
        # subfolers of 'data/train', and indefinitely generate
        # batches of augmented image data
        train_generator = train_datagen.flow_from_directory(
            self.training_path,  # this is the target directory
            target_size=(150, 150),  # all images will be resized to 150x150
            batch_size=self.batch_size,
            class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

        # this is a similar generator, for validation data
        validation_generator = test_datagen.flow_from_directory(
            self.validation_path,
            target_size=(150, 150),
            batch_size=self.batch_size,
            class_mode='binary')

        model.fit_generator(
            train_generator,
            steps_per_epoch=self.num_training_examples // self.batch_size,
            epochs=self.num_epochs,
            validation_data=validation_generator,
            validation_steps=self.num_validation_examples // self.batch_size)
        model.save_weights('second_try.h5')  # always save your weights after training or during training


training_path = 'C:/Users/FTG-006/Desktop/Machine Learning Project/Graph data/graphs_training_data/'
validation_path = 'C:/Users/FTG-006/Desktop/Machine Learning Project/Graph data/graphs_validation_data'

batch_size = 64
num_epochs = 20

test = Model(training_path, validation_path, batch_size, num_epochs)
test.optimise

