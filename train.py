import tensorflow as tf
tf.enable_eager_execution()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

raw_image_dataset = tf.data.TFRecordDataset('dataset_train.tfrecord')

image_feature_description = {
    'label': tf.FixedLenFeature([], tf.int64),
    'image_raw': tf.FixedLenFeature([], tf.string),
}


def _parse_image_function(example_proto):
    return tf.parse_single_example(example_proto, image_feature_description)


parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

