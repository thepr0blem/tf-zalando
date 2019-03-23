import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

from load_data import load_and_process_data
from img_process_and_plots import plot_training_history
from img_process_and_plots import plot_conf_mat, display_errors, plot_samples
from global_vars import *


tf.reset_default_graph()

# Load data and labels
X_train, y_train, X_test, y_test, X_val, y_val = load_and_process_data()

X_full = np.append(np.append(X_train, X_test, axis=0), X_val, axis=0)
y_full = np.append(np.append(y_train, y_test, axis=0), y_val, axis=0)


def conv2d(x, W, b, strides=1):

    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)

    return tf.nn.relu(x)


def maxpool2d(x, k=2):

    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def dropout(x, rate=0.25, is_training=True):

    return tf.layers.dropout(x, rate=rate, training=is_training)


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


def predict_arr(to_predict):
    """Load model and return prediction for "to_predict" array. """

    tf.reset_default_graph()
    inputs, targets, pred, cost, saver, is_train_mode = build_model()
    predict_op = tf.argmax(pred, 1)

    with tf.Session() as sess:

        saver.restore(sess, model_path + model_name)
        prediction = sess.run(predict_op, feed_dict={inputs: to_predict, is_train_mode: False})

    return prediction


def score(X_test, y_test):
    """Calculate accuracy of the model given X, y data"""

    prediction = predict_arr(X_test)

    y_test_vec = np.argmax(y_test, 1)

    return 1 - np.mean(prediction != y_test_vec)


# plot_samples(X_full, y_full, labels_dict)

# train_loss, test_loss, train_accuracy, test_accuracy = train_and_save_model()
# plot_training_history(train_loss, test_loss, train_accuracy, test_accuracy)

print("Test set accuracy: ", score(X_test, y_test))

# y_pred = predict_arr(X_test)
# y_test_cat = np.argmax(y_test, axis=1)

# conf_mat = confusion_matrix(y_test_cat, y_pred)
# labels_list = [labels_dict[i] for i in labels_dict]

# plot_conf_mat(conf_mat, labels_list, normalize=False)

# display_errors(X_test, y_test_cat, y_pred, labels_dict)

