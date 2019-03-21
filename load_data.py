import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from global_vars import *


def load_and_process_data():

    df_train = pd.read_csv(path_train)
    df_test = pd.read_csv(path_test)

    X = df_train.iloc[:, 1:785].values / 255.0
    y = df_train.iloc[:, 0].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
    X_train = np.array(X_train, dtype='f')

    y_train = tf.keras.utils.to_categorical(y_train)
    y_train = np.array(y_train, dtype='f')

    X_val = X_val.reshape(-1, IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
    X_val = np.array(X_val, dtype='f')

    y_val = tf.keras.utils.to_categorical(y_val)
    y_val = np.array(y_val, dtype='f')

    X_test = df_test.iloc[:, 1:785].values / 255.0
    X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
    X_test = np.array(X_test, dtype='f')

    y_test = df_test.iloc[:, 0].values
    y_test = tf.keras.utils.to_categorical(y_test)
    y_test = np.array(y_test, dtype='f')

    return X_train, y_train, X_test, y_test, X_val, y_val



