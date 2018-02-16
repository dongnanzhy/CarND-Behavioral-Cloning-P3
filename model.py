import csv
import cv2
import os
import math
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D, Conv2D, MaxPooling2D

DATA_PATH_UDACITY = './data/data_udacity'
DATA_PATH_SELFGEN = './data/data_0212'
MODEL_PATH = 'model.h5'
BATCH_SIZE = 16


def read_log_file(folder_path):
    samples = []
    with open(os.path.join(folder_path, 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples


def get_generator(samples, batch_size=16):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        shuffle(samples)
        # real batch size = batch_size * 6
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_angle = float(batch_sample[3])
                # create adjusted steering measurements for the side camera images
                correction = 0.2
                left_angle = center_angle + correction
                right_angle = center_angle - correction

                # center image
                filename = batch_sample[0].split('/')[-1]
                data_path = DATA_PATH_SELFGEN if '2018_02_12' in filename else DATA_PATH_UDACITY
                current_path = os.path.join(data_path, 'IMG', filename)
                center_image = cv2.imread(current_path)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_image_flipped = np.fliplr(center_image)
                # left image
                filename = batch_sample[1].split('/')[-1]
                current_path = os.path.join(data_path, 'IMG', filename)
                left_image = cv2.imread(current_path)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                left_image_flipped = np.fliplr(left_image)
                # right image
                filename = batch_sample[2].split('/')[-1]
                current_path = os.path.join(data_path, 'IMG', filename)
                right_image = cv2.imread(current_path)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                right_image_flipped = np.fliplr(right_image)

                images.extend([center_image, center_image_flipped, \
                            left_image, left_image_flipped, \
                            right_image, right_image_flipped])
                angles.extend([center_angle, - center_angle, \
                            left_angle, - left_angle, \
                            right_angle, - right_angle])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def build_model():
    model = Sequential()
    # pre-process
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x / 255.0 - 0.5)))
    # nvidia-model
    model.add(Conv2D(24, (5, 5), strides=(2,2), activation='relu', input_shape=(65,320,3)))
    model.add(Conv2D(36, (5, 5), strides=(2,2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2,2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))
    # compile model
    model.compile(loss='mse', optimizer='adam',  metrics=['accuracy'])
    return model

def visualize_model(model_history):
    ### print the keys contained in the history object
    print(model_history.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()



if __name__ == "__main__":
    ## Read data log filename
    samples_udacity = read_log_file(DATA_PATH_UDACITY)
    samples_selfgen = read_log_file(DATA_PATH_SELFGEN)
    samples = samples_udacity + samples_selfgen

    ## train validation train_test_split
    train_samples, validation_samples = train_test_split(samples, test_size=0.1)

    ## get generater
    train_generator = get_generator(train_samples, batch_size=BATCH_SIZE)
    validation_generator = get_generator(validation_samples, batch_size=BATCH_SIZE)

    ## Build model
    model = build_model()

    ## Train models
    train_history = model.fit_generator(train_generator,
                                        steps_per_epoch= math.ceil(len(train_samples) / BATCH_SIZE),
                                        epochs=30,
                                        validation_data=validation_generator,
                                        validation_steps=math.ceil(len(validation_samples) / BATCH_SIZE))

    ## Save model
    model.save(MODEL_PATH)

    ## Visualize model
    visualize_model(train_history)
