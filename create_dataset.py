import cv2
import glob
import tensorflow as tf

from random import shuffle


def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 147), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def create_tf_data_record(img_paths, img_labels, file_name):
    with tf.python_io.TFRecordWriter(file_name) as writer:
        for index in range(len(img_paths)):
            img = load_image(img_paths[index])

            label_int64_list = tf.train.Int64List(value=[img_labels[index]])
            img_raw_bytes_list = tf.train.BytesList(value=[img.tostring()])
            # img_raw_bytes_list = tf.train.BytesList(value=[tf.compat.as_bytes(img.tostring())])

            feature = {
                'label': tf.train.Feature(int64_list=label_int64_list),
                'image_raw': tf.train.Feature(bytes_list=img_raw_bytes_list)
            }

            features = tf.train.Features(feature=feature)
            example = tf.train.Example(features=features)

            writer.write(example.SerializeToString())


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

create_tf_data_record(train_img_paths, train_img_labels, 'dataset_train.tfrecord')
create_tf_data_record(dev_img_paths, dev_img_labels, 'dataset_dev.tfrecord')
create_tf_data_record(test_img_paths, test_img_labels, 'dataset_test.tfrecord')
