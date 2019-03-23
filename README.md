# CNN Implementation using TensorFlow

The purpose of this project is to develop convolutional neural network for items of clothing (fashion-MNIST dataset) classification. 

### Content of this document 
1. Introduction
2. Data loading, exploring and preprocessing
3. CNN model architecture selection, initialization and training
4. Model evaluation
5. Real world test 
6. Summary

## 1. Introduction 

### 1.1 Technologies and techniques used: 

#### Model architecture 
```Tensorflow```

#### Validation:
Tools provided in ```scikit-learn``` library:
- random division of the sample on training and testing sets
- confusion matrix 

#### Techniques for accuracy improvement:
Estimated accuracy of the classifier: 95.2%. Based on model performance calculated from testing set accuracy. 
Techniques: 
- regularization (via Dropout)  
- gradient descent optimization (Adam Optimizer) 

### 1.2 Project structure 

```
├── models                              # Trained models 
├── images                              # Pictures/visualizations 
├── main.py                             # Code for workflow as presented in README
├── predict_func.py                     # real image classification
├── image_process_and_plots.py          # image processing and plotting charts
├── load_data.py                        # loading data
├── global_vars.py                      # global variables
├── requirments.txt                     # Required libraries
└── README.md                 
```

### 1.3 Dataset overview

Data has been downloaded from [1] in .csv form and stored locally in ```./data``` location.

Observations: 
- training set is provided in form of dataframe with 784 columns (with pixel values) and one additional column with class labels
- training dataset consist of 60,000 examples clothing pictures divided into 10 classes
- the elements of clothing are centered and aligned in terms of the size
- each example is a 28x28 image unfolded to 1x784 vector (data need to be reshaped before feeding into CNN model) 
- pixel values are in gray scale of 0-255 

### 1.4 How to USE

Required technologies are listed in ```requirments.txt``` file.

#### 1.4.1 ```main.py```

To go through all te steps described in this document please use ```main.py``` script 

Imports used in the script with all dependencies 

```python
import tensorflow as tf
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import confusion_matrix
from skimage import io, transform, util
from sklearn.model_selection import train_test_split

from load_data import load_and_process_data
from img_process_and_plots import plot_training_history
from img_process_and_plots import plot_conf_mat, display_errors, plot_samples
from global_vars import *

```

#### 1.4.2 ```predict_func.py```

To leverage already trained model to make your own prediction, use ```predict_func.py``` 

```python
from main import predict_arr

from img_process_and_plots import show_img, load_real_img
from global_vars import *

def predict(photo=samp_photo):
    """This functions classifies and displays given image

        Args:
            photo: string with path to photo for classification
        Returns:
            Plots processed photo as inputted to CNN and predicted class
        """
    test_real = labels_dict[predict_arr(load_real_img(img=photo))[0]]
    show_img(photo, processed=True, real=True, title=test_real)
```
#### 1.4.3 ```global_vars.py```

Global variables and settings are stored in ```global_vars.py```.

```python
IMG_SIZE = 28
IMG_CHANNELS = 1
N_CLASSES = 10
BATCH_SIZE = 128
LR = 1e-3   # Learning rate 
DROPOUT = 0.2

model_path = r"./model/"
model_name = "final"

path_train = r"./data/fashion-mnist_train.csv"
path_test = r"./data/fashion-mnist_test.csv"

samp_photo = r"./images/sample_photo.jpg"
```

## 2. Loading and exploring the data

### 2.1 Loading and preprocessing the data
```python
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
```
```python
X_train, y_train, X_test, y_test, X_val, y_val = load_and_process_data()
```  
Label dictionary 
```python
labels_dict = {0: "T-shirt_top",
               1: "Trouser",
               2: "Pullover",
               3: "Dress",
               4: "Coat",
               5: "Sandal",
               6: "Shirt",
               7: "Sneaker",
               8: "Bag",
               9: "Ankle_boot"}
```  
### 2.2 Exploring data - samples
Plotting classes distribution
```python
def plot_classes_count(y, label_names):
    """Plots histogram with data set classes count.

    Args:
        y: (i x 10) vector with labels, i - number of examples
        label_names: dict with {class: label} structure
    """
    y_classes = [label_names[np.argmax(i)] for i in y]

    sns.set(style="darkgrid")
    count_plot = sns.countplot(y_classes, palette="Blues_d")
    count_plot.set_xticklabels(count_plot.get_xticklabels(), rotation=45, fontsize=9)
```

![Classes](https://github.com/thepr0blem/tf-zalando/blob/master/images/classes_count.PNG) 


Defining function to plot randomly selected, exemplary pictures:
```python
def plot_samples(X, y, labels):
    """Display 3x3 plot with sample images from X, y dataset.

    Args:
        X: (i x IMG_SIZE x IMG_SIZE x 1) array with i examples
        y: (i x 10) vector with labels
        labels: dict with {class: label} structure
    """
    f, pl_arr = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            n = rd.randint(0, X.shape[0])
            pl_arr[i, j].imshow(X[n].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
            pl_arr[i, j].axis('off')
            pl_arr[i, j].set_title(labels[np.argmax(y[n])])
```

![Samples](https://github.com/thepr0blem/tf-zalando/blob/master/images/plot_samples.PNG) 

      
## 3. Building CNN 
### 3.1 Defining CNN architecture

To implement convolutional neural network I used **TensorFlow**.

Layers for final CNN are ordered as follows:

  - **Conv2D** - conv. layer 
    - filters - 32
    - kernel_size - 3 x 3
    - activation - 'relu' 
    - input shape - 4D tensor - (n, 28, 28, 1) - (number of examples, img_size, img_size, no_of_channels) 
    - padding - 'same'
  - **Max_Pooling** - subsampling layer
    - pool_size - (2, 2)
  - **Dropout** - regularization layer
    - dropout_percentage - 20%

  - **Conv2D** - conv. layer 
    - filters - 64
    - kernel_size - 3 x 3
    - activation - 'relu' 
    - padding - 'same'
  - **Max_Pooling** - subsampling layer
    - pool_size - (2, 2)
  - **Dropout** - regularization layer
    - dropout_percentage - 20%
 
   - **Conv2D** - conv. layer 
    - filters - 128
    - kernel_size - 3 x 3
    - activation - 'relu' 
    - padding - 'same'
  - **Max_Pooling** - subsampling layer
    - pool_size - (2, 2)
  - **Dropout** - regularization layer
    - dropout_percentage - 20%
 
  - **Flatten** - flattening input for dense layers input
  - **Dense** - regular dense layer
    - number of neurons - 128
    - activation - 'relu'
  - **Dropout** - regularization layer
    - dropout_percentage - 20%
   
  - **Dense** - final layer
    - units - number of classes
    - activation - 'softmax'
    
```python
def conv2d(x, W, b, strides=1):

    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)

    return tf.nn.relu(x)
```
```python
def maxpool2d(x, k=2):

    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
```
```python
def dropout(x, rate=0.25, is_training=True):

    return tf.layers.dropout(x, rate=rate, training=is_training)
```
```python
def conv_net(x, weights, biases, is_train_mode):

    conv1 = conv2d(x, weights['w_conv1'], biases['b_conv1'])
    conv1 = maxpool2d(conv1, k=2)
    conv1 = dropout(conv1, rate=DROPOUT, is_training=is_train_mode)

    conv2 = conv2d(conv1, weights['w_conv2'], biases['b_conv2'])
    conv2 = maxpool2d(conv2, k=2)
    conv2 = dropout(conv2, rate=DROPOUT, is_training=is_train_mode)

    conv3 = conv2d(conv2, weights['w_conv3'], biases['b_conv3'])
    conv3 = maxpool2d(conv3, k=2)
    conv3 = dropout(conv3, rate=DROPOUT, is_training=is_train_mode)

    fc1 = tf.reshape(conv3, [-1, weights['w_dense'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['w_dense']), biases['b_dense'])
    fc1 = tf.nn.relu(fc1)
    fc1 = dropout(fc1, rate=DROPOUT, is_training=is_train_mode)

    out = tf.add(tf.matmul(fc1, weights['w_out']), biases['b_out'])

    return out
```
```python
def build_model():

    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
    targets = tf.placeholder(dtype=tf.float32, shape=[None, N_CLASSES])

    is_train_mode = tf.placeholder_with_default(True, shape=())

    weights = {
        'w_conv1': tf.get_variable('W0', shape=(3, 3, 1, 32), initializer=tf.contrib.layers.xavier_initializer()),
        'w_conv2': tf.get_variable('W1', shape=(3, 3, 32, 64), initializer=tf.contrib.layers.xavier_initializer()),
        'w_conv3': tf.get_variable('W2', shape=(3, 3, 64, 128), initializer=tf.contrib.layers.xavier_initializer()),
        'w_dense': tf.get_variable('W3', shape=(4 * 4 * 128, 128), initializer=tf.contrib.layers.xavier_initializer()),
        'w_out': tf.get_variable('W6', shape=(128, N_CLASSES), initializer=tf.contrib.layers.xavier_initializer())

    }
    biases = {
        'b_conv1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
        'b_conv2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
        'b_conv3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
        'b_dense': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
        'b_out': tf.get_variable('B4', shape=(10), initializer=tf.contrib.layers.xavier_initializer())

    }

    saver = tf.train.Saver()

    pred = conv_net(inputs, weights, biases, is_train_mode)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=targets))

    return inputs, targets, pred, cost, saver, is_train_mode
```
```python

def train_and_save_model():

    inputs, targets, pred, cost, saver, is_train_mode = build_model()

    optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(targets, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()

    epochs = 50

    with tf.Session() as sess:
        sess.run(init)
        print("Training started.")
        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []

        for i in range(epochs):
            for batch in range(len(X_train)//BATCH_SIZE):
                batch_x = X_train[batch * BATCH_SIZE:min((batch + 1) * BATCH_SIZE, len(X_train))]
                batch_y = y_train[batch * BATCH_SIZE:min((batch + 1) * BATCH_SIZE, len(y_train))]

                sess.run(optimizer, feed_dict={inputs: batch_x, targets: batch_y})

                loss, acc = sess.run([cost, accuracy], feed_dict={inputs: batch_x, targets: batch_y})

            print("Iter " + str(i + 1) + ", Loss= {:.6f}".format(loss) + ", Train Acc= {:.5f}".format(acc))
            print("Optimization for Iter {} Finished!".format(i + 1))

            test_acc, valid_loss = sess.run([accuracy, cost], feed_dict={inputs: X_val, targets: y_val, is_train_mode: False})

            train_loss.append(loss)
            test_loss.append(valid_loss)
            train_accuracy.append(acc)
            test_accuracy.append(test_acc)
            print("Val Accuracy: {:.5f}".format(test_acc))

        saver.save(sess, model_path + model_name)
        print("Model saved.")

        return train_loss, test_loss, train_accuracy, test_accuracy
```
```python
train_loss, test_loss, train_accuracy, test_accuracy = train_and_save_model()
```
Above reports allow us to plot how loss and accuracy changed across 50 epochs for training and validation data. To plot this we use the following function: 

```python
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
```

<img src="https://github.com/thepr0blem/tf-zalando/blob/master/images/acc.PNG" width="400">

<img src="https://github.com/thepr0blem/tf-zalando/blob/master/images/loss.PNG" width="400">

As we can see validation accuracy stopped improving around epoch ~30. 

## 4. Model evaluation
### 4.1 Load model and evaluate on test data set 

```python
def predict_arr(to_predict):
    """Load model and return prediction for "to_predict" array. """

    tf.reset_default_graph()
    inputs, targets, pred, cost, saver, is_train_mode = build_model()
    predict_op = tf.argmax(pred, 1)

    with tf.Session() as sess:

        saver.restore(sess, model_path + model_name)
        prediction = sess.run(predict_op, feed_dict={inputs: to_predict, is_train_mode: False})

    return prediction
```

```python
def score(X_test, y_test):
    """Calculate accuracy of the model given X, y data"""

    prediction = predict_arr(X_test)

    y_test_vec = np.argmax(y_test, 1)

    return 1 - np.mean(prediction != y_test_vec)
```

```python
print("Test set accuracy: ", score(X_test, y_test))
```
```
Test set accuracy:  0.9329
```

### 4.2 Confusion matrix 

Leveraging ```scikit-learn``` modules we can easily build confusion matrix, which will show which classes are the most difficult for the model to distinguish between. 

First, we need to classify test examples and pass the predictions to ```confusion_matrix``` together with true labels.  
```python
y_pred = predict_arr(X_test)
y_test_cat = np.argmax(y_test, axis=1)
```
```python
conf_mat = confusion_matrix(y_test_cat, y_pred)
```

Next, short function for plotting the matrix will be required. I used ```seaborn.heatmap```

```python
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
```
Labels list: 
```python
labels_list = [labels_dict[i] for i in labels_dict]
```
Plotting: 
```python
plot_conf_mat(conf_mat, labels_list, normalize=False)
```

![Confusion matrix](https://github.com/thepr0blem/tf-zalando/blob/master/images/conf_mat.png) 

### 4.3 Display exemplary mistakes 

Define ```display_errors()``` function. 

```python
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
```
```python
display_errors(X_test, y_test_cat, y_pred, labels_dict)
```

![Exemplary errors](https://github.com/thepr0blem/tf-zalando/blob/master/images/exemplary_errors.png) 

## 5. Real world test

What would be our model if we could not use it in a real world environment. I decided to test the model using one of my good old levi's sneakers. Sample photo: 

<img src="https://github.com/thepr0blem/tf-zalando/blob/master/images/sample_photo.jpg" width="400">

To test it on your own you can run ```predict_func.py``` script (already covered in point 1.4.2), remember about changing ```samp_photo``` path variable in ```global_vars.py```. 

Abovementioned predict function uses two additional functions for processing real image and also plot the result. 

```python
def load_real_img(img=samp_photo):

    img = io.imread(img, as_gray=True)
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    img = util.invert(img)

    return img
```

```python
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
```

Let's test it: 

```python
predict(samp_photo)
```
<img src="https://github.com/thepr0blem/tf-zalando/blob/master/images/sample_photo_classification.png" width="400">

Voilà! It worked! 

## 6. Summary 

- estimated model accuracy - 93.3% 
- based on insightful view presented in confusion matrix, we can conclude that the model misclassifies items with similar shape Examples:  
  - shirts confused with T-shirts
  - pullovers confused with shirts 

### References 
[1] https://www.kaggle.com/zalando-research/fashionmnist

[2] https://pythonprogramming.net/cnn-tensorflow-convolutional-nerual-network-machine-learning-tutorial/

[3] https://www.kaggle.com/gpreda/cnn-with-tensorflow-keras-for-fashion-mnist


