import tensorflow as tf
from tensorflow import keras
from random import shuffle

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from helpers import print_length_and_shapes, show_one_image, show_multiple_images, load_fashion_mnist_dataset

train_images, train_labels, test_images, test_labels = load_fashion_mnist_dataset()
class_names = ['T-shirt', 'Shoe']


# Comment out to train on data created with create_dataset. Change input shape to 200, 200

# train_images = np.load('dataset_train_images.npy')
# train_labels = np.load('dataset_train_labels.npy')
# dev_images = np.load('dataset_dev_images.npy')
# dev_labels = np.load('dataset_dev_labels.npy')
# test_images = np.load('dataset_test_images.npy')
# test_labels = np.load('dataset_test_labels.npy')
# class_names = ['Not Activate', 'Activate']

train_images = train_images / 255.0
# dev_images = dev_images / 255.0
test_images = test_images / 255.0


# print_length_and_shapes(train_images, train_labels, dev_images, dev_labels, test_images, test_labels)
show_one_image(train_images[2])
show_multiple_images(train_images, train_labels, class_names)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

model.summary()

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)

# predictions[0]
# np.argmax(predictions[0])
# test_labels[0]


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("Pred.: {} {:2.0f}% (Tru L.: {})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)

plt.show()

print('Making a prediction for the following image')
plt.figure()
plt.imshow(test_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Because keras optimization we need a list of a single image
img = (np.expand_dims(test_images[0], 0))

predictions_single = model.predict(img)

print('single example true label')
print(test_labels[0])
print('single example prediction')
print(predictions_single)
