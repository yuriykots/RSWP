import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def print_length_and_shapes(train_images, train_labels, dev_images, dev_labels, test_images, test_labels):
    print('Debugging (Shapes)')
    print('__________________')
    print('train_images.shape')
    print(train_images.shape)
    print('len(train_labels)')
    print(len(train_labels))
    print('train_labels')
    print(train_labels)

    print('dev_images.shape')
    print(dev_images.shape)
    print('len(dev_labels)')
    print(len(dev_labels))
    print('dev_labels')
    print(dev_labels)

    print('test_images.shape')
    print(test_images.shape)
    print('len(test_labels)')
    print(len(test_labels))
    print('test_labels')
    print(train_labels)


def show_one_image(img):
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def show_multiple_images(images, labels, class_names):
    plt.figure(figsize=(10, 10))

    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])

    plt.show()
