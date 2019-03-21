from skimage import io, transform, util
import matplotlib.pyplot as plt
import random as rd
import numpy as np
import seaborn as sns

from global_vars import *


def load_real_img(img=samp_photo):

    img = io.imread(img, as_gray=True)
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    img = util.invert(img)

    return img


def show_img(img=samp_photo, processed=False, real=False, title=""):

    if real:

        if processed:
            img = io.imread(img, as_gray=True)
            img = transform.resize(img, (IMG_SIZE, IMG_SIZE))
            img = (img - 1) * (-1)

            fig = plt.imshow(img, cmap="gray")

        else:
            img = io.imread(img)
            fig = plt.imshow(img, cmap="gray")

    else:
        img = img.reshape((IMG_SIZE, IMG_SIZE))

        fig = plt.imshow(img, cmap="gray")

    plt.axis('off')
    plt.title(title)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)


def plot_training_history(train_loss, test_loss, train_accuracy, test_accuracy):

    plt.plot(range(len(train_loss)), train_loss, 'b', label='Training loss')
    plt.plot(range(len(train_loss)), test_loss, 'r', label='Test loss')
    plt.title('Training and Test loss')
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend()
    plt.figure()
    plt.show()

    plt.plot(range(len(train_loss)), train_accuracy, 'b', label='Training Accuracy')
    plt.plot(range(len(train_loss)), test_accuracy, 'r', label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend()
    plt.figure()
    plt.show()


def plot_samples(X, y, labels):
    """Display 3x3 plot with sample images from X, y dataset.

    Args:
        X: (i x IMG_SIZE x IMG_SIZE x 1) array with i examples
        y: (i x 1) vector with labels
        n: dict with {class: label} structure

    Returns:
        Displays n-th example and returns class label.
    """
    f, pl_arr = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            n = rd.randint(0, X.shape[0])
            pl_arr[i, j].imshow(X[n].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
            pl_arr[i, j].axis('off')
            pl_arr[i, j].set_title(labels[np.argmax(y[n])])


def plot_conf_mat(conf_mat, label_list, normalize=False):
    """Plots confusion matrix"""

    if normalize:
        conf_mat = conf_mat.astype(float) / conf_mat.sum(axis=1)[:, np.newaxis]

    fmt = '.2f' if normalize else 'd'
    sns.heatmap(conf_mat, annot=True, fmt=fmt,
                cmap="Blues", cbar=False, xticklabels=label_list,
                yticklabels=label_list, robust=True)
    plt.title('Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def display_errors(X, y_true, y_pred, labels):
    """This function shows 9 wrongly classified images (randomly chosen)
    with their predicted and real labels """

    errors = (y_true - y_pred != 0).reshape(len(y_pred), )

    X_errors = X[errors]
    y_true_errors = y_true[errors]
    y_pred_errors = y_pred[errors]

    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True)
    for i in range(3):
        for j in range(3):
            n = rd.randint(0, len(X_errors))
            ax[i, j].imshow(X_errors[n].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
            ax[i, j].set_title("Predicted label :{}\nTrue label :{}"
                               .format(labels[y_pred_errors[n]], labels[y_true_errors[n]]))


def plot_classes_count(y_full, label_names):

    y_classes = [label_names[np.argmax(i)] for i in y_full]

    sns.set(style="darkgrid")
    count_plot = sns.countplot(y_classes, palette="Blues_d")
    count_plot.set_xticklabels(count_plot.get_xticklabels(), rotation=45, fontsize=9)
