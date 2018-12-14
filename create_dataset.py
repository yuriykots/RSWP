import glob
from random import shuffle

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
