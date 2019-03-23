import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from global_vars import *


def load_and_process_data():

    # Load data from csv file
    df_train = pd.read_csv(path_train)
    df_test = pd.read_csv(path_test)

    # Extracting images and labels
    X = df_train.iloc[:, 1:785].values / 255.0
    y = df_train.iloc[:, 0].values

    # Split on training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    # Reshaping training set to 4-D array required to CNN input
    X_train = np.array(X_train.reshape(-1, IMG_SIZE, IMG_SIZE, IMG_CHANNELS), dtype='f')

    # One hot encoding of training labels
    y_train = np.array(tf.keras.utils.to_categorical(y_train), dtype='f')

    # Reshaping validation set to 4-D array required to CNN input
    X_val = np.array(X_val.reshape(-1, IMG_SIZE, IMG_SIZE, IMG_CHANNELS), dtype='f')

    # One hot encoding of validation labels
    y_val = np.array(tf.keras.utils.to_categorical(y_val), dtype='f')

    # Extracting images and labels from testing data set
    X_test = df_test.iloc[:, 1:785].values / 255.0
    y_test = df_test.iloc[:, 0].values

    # Reshaping testing set to 4-D array required to CNN input
    X_test = np.array(X_test.reshape(-1, IMG_SIZE, IMG_SIZE, IMG_CHANNELS), dtype='f')

    # One hot encoding of validation test set labels
    y_test = np.array(tf.keras.utils.to_categorical(y_test))

    return X_train, y_train, X_test, y_test, X_val, y_val



