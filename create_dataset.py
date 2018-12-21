import cv2
import glob
import numpy as np
from random import shuffle


def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 147), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def create_numpy_array_data_record(paths, labels, file_name):
    data_images = np.array([np.array(load_image(img_path)) for img_path in paths])
    data_labels = np.asarray(labels)
    data_images.dump('{}_images.npy'.format(file_name))
    data_labels.dump('{}_labels.npy'.format(file_name))


img_paths = glob.glob('images/*/*.jpg')
img_labels = list(0 if 'false' in address else 1 for address in img_paths)
img_paths_and_labels = list(zip(img_paths, img_labels))
shuffle(img_paths_and_labels)
img_paths, img_labels = zip(*img_paths_and_labels)

# Train 60%, Dev 20%, Test 20%
train_img_paths = img_paths[0: int(0.6*len(img_paths))]
train_img_labels = img_labels[0: int(0.6*len(img_labels))]
dev_img_paths = img_paths[int(0.6*len(img_paths)):int(0.8*len(img_paths))]
dev_img_labels = img_labels[int(0.6*len(img_labels)):int(0.8*len(img_labels))]
test_img_paths = img_paths[int(0.8*len(img_paths)):]
test_img_labels = img_labels[int(0.8*len(img_labels)):]

create_numpy_array_data_record(train_img_paths, train_img_labels, 'dataset_train')
create_numpy_array_data_record(dev_img_paths, dev_img_labels, 'dataset_dev')
create_numpy_array_data_record(test_img_paths, test_img_labels, 'dataset_test')
