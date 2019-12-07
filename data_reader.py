import random
import numpy as np
import os
import cv2
from os import scandir


def data_reader(input_dir):
    """
    Read images from input_dir then shuffle them
    Args:
    input_dir: string, path of input dir, e.g., /path/to/dir
    Returns:
    file_paths: list of strings
    """
    file_paths = []
    file_labels = []
    label = -1
    for img_fold in scandir(input_dir):
        label = label + 1
        # print(img_fold, label)
        for img_file in scandir(img_fold.path):
            file_paths.append(img_file.path)
            file_labels.append(label)
    return file_paths, file_labels
def get_single_img(path,label=1,image_width=256,image_height=256):
    image=cv2.imread(path)
    image=cv2.resize(image,(image_width,image_height))
    feed_img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    feed_img=np.array(feed_img)/127.5-1.
    return [feed_img],[label],image



def get_source_batch(batch_size, image_width, image_height, source_dir=""):
    file_paths, file_labels = data_reader(source_dir)
    max_size = len(file_paths)
    if batch_size > max_size:
        batch_size = max_size
    idx_list = random.sample(range(0, max_size), batch_size)
    files = []
    labels = []
    images = []
    oimgs=[]
    #file_dirs=[]
    if batch_size == 0:
        batch_size = max_size
        for i in range(batch_size):
            files.append(file_paths[i])
            labels.append(file_labels[i])
            image = cv2.imread(os.path.join(file_paths[i]))
            image = cv2.resize(image, (image_width, image_height))
            oimgs.append(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image) / 127.5 - 1.
            images.append(image)
    else:
        for i in range(batch_size):
            for j in range(max_size):
                if idx_list[i] == j:
                    files.append(file_paths[j])
                    labels.append(file_labels[j])
                    image = cv2.imread(os.path.join(file_paths[j]))
                    image = cv2.resize(image, (image_width, image_height))
                    oimgs.append(image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = np.array(image) / 127.5 - 1.
                    images.append(image)
    return images, idx_list, len(file_paths), labels,oimgs,files


def get_target_batch(batch_size, image_width, image_height, target_dir=""):
    #print(target_dir)
    file_paths, file_labels = data_reader(target_dir)
    max_size = len(file_paths)
    if batch_size > max_size:
        batch_size = max_size - 1
    idx_list = random.sample(range(0, max_size), batch_size)
    files = []
    labels = []
    images = []
    oimgs = []
    if batch_size == 0:
        batch_size = max_size
        for i in range(batch_size):
            files.append(file_paths[i])
            labels.append(file_labels[i])
            image = cv2.imread(os.path.join(file_paths[i]))
            image = cv2.resize(image, (image_width, image_height))
            oimgs.append(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image) / 127.5 - 1.
            images.append(image)
    else:
        for i in range(batch_size):
            for j in range(max_size):
                if idx_list[i] == j:
                    files.append(file_paths[j])
                    labels.append(file_labels[j])
                    image = cv2.imread(os.path.join(file_paths[j]))
                    image = cv2.resize(image, (image_width, image_height))
                    oimgs.append(image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = np.array(image) / 127.5 - 1.
                    images.append(image)
    return images, idx_list, len(file_paths), labels,oimgs,files


if __name__ == '__main__':
    UC_name = "crescent"
    source_path = "/home/amax/AijiaLi/FCM/datasets/" + UC_name
    target_path = "/home/amax/AijiaLi/FCM/target/train"
    source_paths = get_source_batch(0, 244, 244, source_dir=source_path)
    # target_paths = get_target_batch(0, 224, 224, target_dir=target_path)
    print(source_paths)
    # print(target_paths)
