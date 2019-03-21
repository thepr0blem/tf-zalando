UNDER CONSTRUCTION

# CNN Implementation using TensorFlow

The purpose of this project is to develop convolutional neural network for written characters classification. 

### Content of this document 
1. Introduction
2. Data loading, exploring and preprocessing
3. CNN model architecture selection, initialization and training
4. Model evaluation

## 1. Introduction 

### 1.1 Technologies and techniques used: 

#### Model architecture 
```Keras``` library (framework based on ```Tensorflow```) 

#### Validation:
Tools provided in ```scikit-learn``` library:
- random division of the sample on training and testing sets
- confusion matrix 

#### Techniques for accuracy improvement:
Estimated accuracy of the classifier: 95.2%. Based on model performance calculated from testing set accuracy. 
Techniques: 
- regularization (via Dropout) 
- hyperparameter tuning (via Random Search) 
- early stopping 
- learning rate reduction (via ReduceLROnPlateau) 
- gradient descent optimization ("adam", "adamax", "nadam", "RMSProp")
- image augmentation (rotation and shift) - tested, but not used - to perform augmentation please run ```./src/augmentation.py``` script

### 1.2 Project structure 

```
├── archive                 # Old models 
├── data                    # Data sets
├── logs                    # Training history logs 
├── models                  # Trained models 
├── pics                    # Pictures/visualizations 
├── src                     # Source files 
├── workflow.py             # Code for workflow as presented in README
├── predict.py              # Function for new data classification 
├── requirments.txt         # Required libraries
└── README.md                 
```

### 1.3 Dataset overview

**NOTE:** In the original data set there were 36 classes and one of them (class #30) had only one example.
          This class was overlapping with class #14 (both were letter "N"), single example was renamed #30 -> #14. Data set with this    change was saved as ```train_fix.pkl```

Observations: 
- training set is provided in form of numpy arrays with 3,136 columns (with pixel values) and one additional vector with class labels
- training dataset consist of 30,134 examples of written characters divided into 35 classes - 10 digits and 25 letters
- the characters are centered and aligned in terms of the size
- the classes are not in order (letters and digits are mixed) 
- there is no data/class for letter "X" 
- each example is a 56x56 image unfolded to 1x3136 vector (data need to be reshaped before feeding into CNN model) 
- pixel values are binary (0 - black / 1 - white) 

### 1.4 How to USE

Required technologies are listed in ```requirments.txt``` file.

#### 1.4.1 ```workflow.py```

To go through all te steps described in this document please use ```workflow.py``` script 

Imports used in the script with all dependencies 

```python
import numpy as np
import pickle
import seaborn as sns
import random as rd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from keras.models import load_model

from src import modelling as mod
from src import visualization as vis
import predict as pred
```

#### 1.4.2 ```predict.py```

To leverage already trained model to make your own prediction, use ```predict.py``` 

```python
def predict(input_data):
    """This functions classifies given input

    Args:
        input_data: (n x 3136) array, where n - # of examples

    Returns:
        output_data: (n x 1) array with class labels
    """
    model_in = load_model(r"./models/CNN_FF_3.h5")

    prediction = model_in.predict(input_data.reshape(input_data.shape[0], 56, 56, 1))

    output_data = prediction.argmax(axis=1).reshape(len(prediction), 1)

    return output_data
```

## 2. Loading and exploring the data

### 2.1 Loading the data
```python
data_dir = r'./data/train_fix.pkl'

with open(data_dir, 'rb') as f:
    data = pickle.load(f)

X, y = data
```
Label dictionary 
```python
labels = {0: "6", 1: "P", 2: "O", 3: "V", 4: "W", 5: "3", 6: "A", 
          7: "8", 8: "T", 9: "I", 10: "0", 11: "9", 12: "H", 13: "R", 
          14: "N", 15: "7", 16: "K", 17: "L", 18: "G", 19: "4", 20: "Y", 
          21: "C", 22: "E", 23: "J", 24: "5", 25: "1", 26: "S", 27: "2", 
          28: "F", 29: "Z", 31: "Q", 32: "M", 33: "B", 34: "D", 35: "U"}
```  
### 2.2 Exploring data - samples
Plotting classes distribution
```python
y_classes = sorted([labels[i] for i in y.reshape(y.shape[0], )])

sns.set(style="darkgrid")
count_plot = sns.countplot(y_classes, palette="Blues_d")

plt.show()
```

IMG CLASSES COUNT


Defining function to plotting randomly selected, exemplary pictures:
```python
def plot_samples(X, y, labels):
    """Display 3x3 plot with sample images from X, y dataset.

    Args:
        X: (i x j) array with i examples (each of them j features)
        y: (i x 1) vector with labels
        n: dict with {class: label} structure

    Returns:
        Displays n-th example and returns class label.
    """
    f, plarr = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            n = rd.randint(0, X.shape[0])
            plarr[i, j].imshow(X[n].reshape(56, 56), cmap='gray')
            plarr[i, j].axis('off')
            plarr[i, j].set_title(labels[y[n][0]])
```
```
vis.plot_samples(X, y)
```

IMG PLOT SAMPLES

### 2.3 Preprocessing the data
#### 2.3.1 Reshaping 

```python
n_cols = X.shape[1]
img_size = int(np.sqrt(n_cols))
no_of_classes = len(np.unique(y, return_counts=True)[0])

X_cnn = X.reshape(X.shape[0], img_size, img_size, 1)
```

#### 2.3.2 Split into training and testing set 
The data has been split into two data sets in 80:20 proportion.  
```python
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_cnn, y, test_size=0.2, random_state=42)
```

#### 2.3.3 Label encoding 

```python
y_train_cat_cnn = to_categorical(y_train_cnn)
y_test_cat_cnn = to_categorical(y_test_cnn)
```
      
## 3. Building CNN 
### 3.1 Defining CNN architecture

To implement convolutional neural network I used **Keras** API (which is user friendly framework built on top of Tensorflow). I used Sequential model which is ordered hierarchy of layers. The architecture has been chosen based on research and articles listed in references ([1] in this case). 

Alternative approach that could be taken: adding additional functionality to ```run_random_search()``` function which would consider different numbers of sets of layers. 

Layers for final CNN are ordered as follows (selection of hyperparameters is presented in the following steps):

  - **Conv2D** - conv. layer 
    - filters - 40
    - kernel_size - 5 x 5
    - activation - 'relu' 
    - input shape - 4D tensor - (n, 56, 56, 1), where (number of examples, img_size, img_size, no_of_channels) 
    - padding - 'same'
  - **Conv2D** - conv. layer 
    - filters - 40
    - kernel_size - 5 x 5
    - activation - 'relu' 
    - padding - 'same'
  - **Max_Pooling** - subsampling layer
    - pool_size - (3, 3)
  - **Dropout** - regularization layer
    - dropout_percentage - 30%

  - **Conv2D** - conv. layer 
    - filters - 40
    - kernel_size - 5 x 5
    - activation - 'relu' 
    - padding - 'same'
  - **Conv2D** - conv. layer 
    - filters - 40
    - kernel_size - 5 x 5
    - activation - 'relu' 
    - padding - 'same'
  - **Max_Pooling** - subsampling layer
    - pool_size - (3, 3)
  - **Dropout** - regularization layer
    - dropout_percentage - 30%
 
  - **Flatten** - flattening input for dense layers input
  - **Dense** - regular dense layer
    - number of neurons - 512
    - activation - 'relu'
  - **Dropout** - regularization layer
    - dropout_percentage - 30%
   
  - **Dense** - final layer
    - units - number of classes
    - activation - 'softmax'
    
```python
def create_model(X, y, it=1, no_of_filters=32, kern_size=3,
                 max_p_size=3, drop_perc_conv=0.3, drop_perc_dense=0.2,
                 dens_size=128, val_split_perc=0.1, no_of_epochs=5,
                 optimizer="adam", random_search=False, batch_size=64):
    """Creates an architecture, train and saves CNN model.

    Returns:
        Dictionary with training report history.
    """

    y_train_cat = to_categorical(y)

    model = Sequential()

    model.add(Conv2D(no_of_filters,
                     kernel_size=(kern_size, kern_size),
                     activation='relu',
                     input_shape=(56, 56, 1),
                     padding='same'))

    model.add(Conv2D(no_of_filters,
                     kernel_size=(kern_size, kern_size),
                     activation='relu',
                     padding='same'))
    model.add(MaxPooling2D((max_p_size, max_p_size)))
    model.add(Dropout(drop_perc_conv))

    model.add(Conv2D(no_of_filters,
                     kernel_size=(kern_size, kern_size),
                     activation='relu',
                     padding='same'))
    model.add(Conv2D(no_of_filters,
                     kernel_size=(kern_size, kern_size),
                     activation='relu',
                     padding='same'))
    model.add(MaxPooling2D((max_p_size, max_p_size)))
    model.add(Dropout(drop_perc_conv))

    model.add(Flatten())

    model.add(Dense(dens_size, activation='relu'))
    model.add(Dropout(drop_perc_dense))

    model.add(Dense(36, activation='softmax'))

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping_monitor = EarlyStopping(patience=5)
    rlrop = ReduceLROnPlateau(monitor='val_acc', factor=0.5, 
                              patience=3, verbose=1, min_lr=0.00001)

    history = model.fit(X,
                        y_train_cat,
                        validation_split=val_split_perc,
                        epochs=no_of_epochs,
                        callbacks=[early_stopping_monitor, rlrop],
                        batch_size=batch_size)

    history_dict = history.history

    if random_search:

        np.save(r"./models/random_search/hist/history_dict_{}.npy".format(it), history_dict)
        model.save(r"./models/random_search/models/CNN_{}.h5".format(it))

    else:

        np.save(r"./logs/history_dict_{}.npy".format(it), history_dict)
        model.save(r"./models/CNN_FF_{}.h5".format(it))

    return history_dict
```

## 4. Model evaluation
### 4.1 Load model and evaluate on test data set 

```python
model_1 = keras.models.load_model(r"./models/CNN_FF_1.h5")
model_2 = keras.models.load_model(r"./models/CNN_FF_2.h5")
model_3 = keras.models.load_model(r"./models/CNN_FF_3.h5")
```

```python
score_1 = model_1.evaluate(X_test_cnn, y_test_cat_cnn, batch_size=64)
score_2 = model_2.evaluate(X_test_cnn, y_test_cat_cnn, batch_size=64)
score_3 = model_3.evaluate(X_test_cnn, y_test_cat_cnn, batch_size=64)
```

```python
print("Model_1: val_acc - {}".format(np.round(score_1[1] * 100, 2)),
      "\nModel_2: val_acc - {}".format(np.round(score_2[1] * 100, 2)),
      "\nModel_3: val_acc - {}".format(np.round(score_3[1] * 100, 2)))
```
```
Model_1: val_acc - 94.72% 
Model_2: val_acc - 95.16% 
Model_3: val_acc - 95.19%
```

Highest accuracy on test set has been identified for **model_3** which is considered as final from now. 
Its accuracy is estimated on **95.2%**.

### 4.2 Confusion matrix 

Leveraging ```scikit-learn``` modules we can easily build confusion matrix, which will show which classes are the most difficult for the model to distinguish between. 

First, we need to classify test examples and pass the predictions to ```confusion_matrix``` together with true labels.  
```python
y_pred = pred.predict(X_test_cnn)
```
```python
conf_mat = confusion_matrix(y_test_cnn, y_pred)
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
labels_list = [labels[i] for i in range(36)]
```
Plotting: 
```python
vis.plot_conf_mat(conf_mat, labels_list, normalize=False)
```

IMG CONF MATRIX 

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
            ax[i, j].imshow(X_errors[n].reshape(56, 56), cmap='gray')
            ax[i, j].set_title("Predicted label :{}\nTrue label :{}"
                               .format(labels[y_pred_errors[n][0]], labels[y_true_errors[n][0]]))
```
```python
vis.display_errors(X_test_cnn, y_test_cnn, y_pred, labels)
```

SAMPLE ERRORS

## Summary 

- estimated model accuracy - 95.2% 
- based on insightful view presented in confusion matrix, we can conclude that the model misclassifies characters with similar shape Examples:  
  - "o" vs "0" - 78/82 examples of "0" classified as "o"
  - "i" vs "1" - 55/60 examples of "i" classified as "1"
  - "z' vs "2" - 17x confused with each other
  - "v" vs "u" - 14x confused with each other
- errors made by classifier are easier to understand if we take a look at exemplary errors in section 4.3. Some of those probably could be also misclassified by human eye (example "U" vs "O" 
- during the development process data augmentation (via small 10 degree rotation and 0.1 relative position translation) was also considered and tested. However, the same model had 93.2% accuracy on test set therefore the solution was not adapted 

### References 
LINK TO DATA SET 

