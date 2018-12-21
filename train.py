import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


train_images = np.load('dataset_train_images.npy')
train_labels = np.load('dataset_train_labels.npy')
dev_images = np.load('dataset_dev_images.npy')
dev_labels = np.load('dataset_dev_labels.npy')
test_images = np.load('dataset_test_images.npy')
test_labels = np.load('dataset_test_labels.npy')

class_names = ['False', 'True']

train_images = train_images / 255.0
dev_images = dev_images / 255.0
test_images = test_images / 255.0

# Debugging (Shapes)
# print('Debugging (Shapes)')
# print('train_images.shape')
# print(train_images.shape)
# print('len(train_labels)')
# print(len(train_labels))
# print('train_labels')
# print(train_labels)
# print('test_images.shape')
# print(test_images.shape)
# print('len(test_labels)')
# print(len(test_labels))

# Debugging (One image)
# print('Debugging (One image)')
# plt.figure()
# plt.imshow(train_images[2])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# Debugging (Multiple images)
# print('Debugging (Multiple images)')
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
#
# plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(147, 256, 3)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)
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


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(2), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


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
