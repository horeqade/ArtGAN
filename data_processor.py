import glob
import math
import os
from os.path import join
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def normalize(arr):
    """
    Linear normalization of image
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[..., i].min()
        maxval = arr[..., i].max()
        if minval != maxval:
            arr[..., i] -= minval
            arr[..., i] *= (255.0 / (maxval - minval))
    return arr


def get_data(path: str, missing_folders: List[str] = None, folders: List[str] = None):
    """

    Args:
        path: path to general folder, that contains folders with images of same style (class)
        missing_folders: folders that won't be processed
        folders: folders that will be processed and only that folders

    Returns:
        Tuple: List of paths to images, number of classes, class_number_dict

    """

    if folders is not None:
        styles_folder = folders
    else:
        styles_folder = os.listdir(path=path)
    if missing_folders is not None:
        styles_folder = [style for style in styles_folder if style not in missing_folders]

    # Show classes for training
    for i, style in enumerate(styles_folder):
        print(i, style)

    class_num_dict = {}
    for i, style in enumerate(styles_folder):
        class_num_dict[style] = i
    num_styles = len(styles_folder)

    data = []
    for i in range(num_styles):
        data += glob.glob(join(path, styles_folder[i], '*'))
    return data, num_styles, class_num_dict


def get_images(indices: List[int], data: List[str], batch_size: int, img_shape: Tuple[int, int, int]):
    """
    Function allow get batch of images for training
    Args:
        indices: Indices of images
        data: Data from "get_data" function. List of paths to imgs
        batch_size: batch size
        img_shape: Resolution of image with 3-d dim - number of colours (1 for grayscale, 3 for rgb)

    Returns:

    """
    img_rows, img_cols, img_channels = img_shape

    color_mode = 'rgb'
    if img_channels == 1:
        color_mode = 'grayscale'

    x_train = np.zeros((batch_size, img_rows, img_cols, img_channels))

    if color_mode == 'grayscale':
        for i in range(batch_size):
            temp_img = cv2.imread(data[indices[i]], 0)
            x_train[i, :, :, 0] = temp_img
    else:
        for i in range(batch_size):
            temp_img = cv2.imread(data[indices[i]])
            x_train[i] = temp_img

    x_train = (x_train - 127.5) / 127.5
    return x_train


def get_images_classes(batch_size: int, data: List[str], classes: dict, img_shape: Tuple[int],
                       batch_num: int, channel_first: bool = False):
    """
    Function not using
    Args:
        batch_size: len of minibatch
        data: list of paths to images
        classes: dict for mapping classes and number of class
        img_shape: Resolution of image with 3-d dim - number of colours (1 for grayscale, 3 for rgb)
        indexes: indexes of images

    Returns:

    """
    if channel_first:
        img_channels, img_rows, img_cols = img_shape
        X_train = np.zeros((batch_size, img_channels, img_rows, img_cols))
    else:
        img_rows, img_cols, img_channels = img_shape
        X_train = np.zeros((batch_size, img_rows, img_cols, img_channels))
    y_labels = np.zeros(batch_size)

    for i in range(batch_size):
        # img_path = data[choice_arr[i]]
        img_path = data[batch_num * batch_size + i]
        temp_img = Image.open(img_path)

        if temp_img.mode != 'RGB':
            temp_img = temp_img.convert('RGB')

        style = img_path.split('\\')[-2]
        temp_img = np.array(temp_img)
        if channel_first:
            temp_img = np.transpose(temp_img, (2, 0, 1))
        temp_img = (temp_img - 127.5) / 127.5
        X_train[i] = temp_img
        y_labels[i] = classes[style]

    return X_train, y_labels


def get_one_image(batch_size: int, data: List[str], classes: dict, img_shape: Tuple[int], img_idx: int = 0,
                  channel_first: bool = False):
    if channel_first:
        img_channels, img_rows, img_cols = img_shape
        X_train = np.zeros((batch_size, img_channels, img_rows, img_cols))
    else:
        img_rows, img_cols, img_channels = img_shape
        X_train = np.zeros((batch_size, img_rows, img_cols, img_channels))
    y_labels = np.zeros(batch_size)
    for i in range(batch_size):
        # img_path = data[choice_arr[i]]
        img_path = data[img_idx]
        temp_img = Image.open(img_path)
        if temp_img.mode != 'RGB':
            temp_img = temp_img.convert('RGB')

        style = img_path.split('\\')[-2]
        temp_img = np.array(temp_img)
        if channel_first:
            temp_img = np.transpose(temp_img, (2, 0, 1))
        temp_img = (temp_img - 127.5) / 127.5
        X_train[i] = temp_img
        y_labels[i] = classes[style]

    return X_train, y_labels


def get_images_one_class(batch_size, data, class_target, img_shape):
    img_rows, img_cols, img_channels = img_shape
    X_train = np.zeros((batch_size, img_rows, img_cols, img_channels))
    y_label = np.zeros(batch_size)

    for i in range(4):
        for j in range(4):
            temp_img = cv2.imread(data[i][j])

            X_train[4 * i + j] = temp_img
            y_label[4 * i + j] = i

    X_train = (X_train - 127.5) / 127.5
    return X_train, y_label


def combine_images(generated_images: np.array):
    """
    Function create big table (square) of images
    Args:
        generated_images: Output of generator

    Returns:
        np.array:: image where all input images concatenated
    """
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:]
    image = np.zeros((height * shape[0], width * shape[1], shape[2]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = \
            img[:, :, :, ]
    return image
