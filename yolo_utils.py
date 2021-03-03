import os
import sys
import numpy as np
import tensorflow as tf
import random
import math
import warnings
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

IMG_WIDTH = 384
IMG_HEIGHT = 384
IMG_CHANNELS = 3

TRAIN_PATH = 'stage1_train'
TEST_PATH = 'stage1_test'

MAX_BB_CNT = 2


def store_bounding_boxes(img, train_id, mask_id, rotby_90):
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh.astype(np.uint8), 1, 2)
    # print(contours)
    cnt = contours[0]
    # print(cnt)
    # print(cv2.boundingRect(cnt)   )
    x, y, w, h = cv2.boundingRect(cnt)

    x = x * (IMG_WIDTH / img.shape[1])
    w = w * (IMG_WIDTH / img.shape[1])
    y = y * (IMG_WIDTH / img.shape[0])
    h = h * (IMG_WIDTH / img.shape[0])

    if (x > IMG_WIDTH - 1):
        x = IMG_WIDTH - 1
    if (y > IMG_HEIGHT - 1):
        y = IMG_HEIGHT - 1
    if (x + w > IMG_WIDTH - 1):
        w = IMG_WIDTH - 1 - x
    if (y + h > IMG_HEIGHT - 1):
        h = IMG_HEIGHT - 1 - y

    bbdict = {"train_id": train_id, "mask_id": mask_id, "rotby_90": rotby_90, "x": x, "y": y,
              "w": w, "h": h}
    return bbdict


def get_grid_info(tr_id, rotby_90):
    df = bboxes.loc[(bboxes.train_id == tr_id) & (bboxes.rotby_90 == rotby_90),
         'grid_row':'box_area']
    df.drop(['grid_center_x', 'grid_center_y', 'box_center_x', 'box_center_y', ], axis=1,
            inplace=True)
    df = df.sort_values(['grid_column', 'grid_row', 'box_area'], ascending=False)
    # print(len(df))
    global mask_count
    mask_count += len(df)
    label_info = np.zeros(shape=(GRID_DIM, GRID_DIM, MAX_BB_CNT, 5), dtype=np.float32) + 0.000001

    for ind, row in df.iterrows():
        i = int(row[0])
        j = int(row[1])
        for b in range(MAX_BB_CNT):
            if (label_info[i, j, b][4] != 1.0):
                label_info[i, j, b] = np.array(row[2:7])
                break
    return label_info


def normalize_yolo_loss(processed_logits, lambda_coords, lambda_noobj):
    yolo_loss = tf.reduce_sum(tf.squared_difference(labels, processed_logits), axis=0)
    yolo_loss = tf.reduce_sum(yolo_loss, axis=0)
    yolo_loss = tf.reduce_sum(yolo_loss, axis=0)
    yolo_loss = tf.reduce_sum(yolo_loss, axis=0)

    yolo_loss = tf.stack([tf.multiply(lambda_coords, yolo_loss[0]),
                          tf.multiply(lambda_coords, yolo_loss[1]),
                          yolo_loss[2],
                          yolo_loss[3],
                          tf.multiply(lambda_noobj, yolo_loss[4])])
    yolo_loss = tf.reduce_sum(yolo_loss)

    return yolo_loss


def get_labels(counts, rotations):
    grid_info = np.zeros(shape=(BATCH_SIZE, GRID_DIM, GRID_DIM, MAX_BB_CNT, 5), dtype=np.float32)
    for i, c in enumerate(counts):
        tr_id = train_ids_df.loc[c, 'id_']
        grid_info[i] = get_grid_info(tr_id, rotations[i])
    grid_info = np.reshape(grid_info, newshape=[BATCH_SIZE, GRID_DIM, GRID_DIM, MAX_BB_CNT, 5])
    return grid_info


def get_images(counts, rotations):
    images = np.zeros(shape=(BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), dtype=np.uint8)
    for i, c in enumerate(counts):
        images[i] = np.rot90(train_images[c], rotations[i])
    return images


def next_batch():
    rotations = []
    rand_counts = []
    for i in range(BATCH_SIZE):
        rotations.append(random.randint(0, 3))
        rand_counts.append(random.randint(0, 669))
    return get_images(rand_counts, rotations), get_labels(rand_counts, rotations)


from tensorflow.python.framework import ops

ops.reset_default_graph()
X = tf.placeholder(tf.float32, [None, IMG_WIDTH, IMG_HEIGHT, 3])
Y_ = tf.placeholder(tf.float32, [None, GRID_DIM, GRID_DIM, MAX_BB_CNT, 5])
lr = tf.placeholder(tf.float32)


def process_logits(logits, name=None):
    net = tf.reshape(logits, [-1, GRID_DIM * 1, GRID_DIM * 1, MAX_BB_CNT * 5 * 16, 1])
    net = tf.layers.average_pooling3d(net, [1, 1, 16], [1, 1, 16], padding="valid")

    net = tf.reshape(net, [-1, GRID_DIM * GRID_DIM * MAX_BB_CNT, 5])  # GRID_DIM = 12
    net = tf.transpose(net, [1, 2, 0])

    logits_tensor = tf.map_fn(lambda x:
                              tf.stack([
                                  tf.tanh(x[0]),
                                  tf.tanh(x[1]),
                                  tf.sqrt(tf.sigmoid(x[2])),
                                  tf.sqrt(tf.sigmoid(x[3])),
                                  tf.sigmoid(x[4])
                              ]), net)

    logits_tensor = tf.transpose(logits_tensor, [2, 0, 1])
    logits_tensor = tf.reshape(logits_tensor, [-1, GRID_DIM, GRID_DIM, MAX_BB_CNT, 5])

    return logits_tensor


def normalize_yolo_loss(processed_logits, lambda_coords, lambda_noobj):
    yolo_loss = tf.reduce_sum(tf.squared_difference(labels, processed_logits), axis=0)
    yolo_loss = tf.reduce_sum(yolo_loss, axis=0)
    yolo_loss = tf.reduce_sum(yolo_loss, axis=0)
    yolo_loss = tf.reduce_sum(yolo_loss, axis=0)

    yolo_loss = tf.stack([tf.multiply(lambda_coords, yolo_loss[0]),
                          tf.multiply(lambda_coords, yolo_loss[1]),
                          yolo_loss[2],
                          yolo_loss[3],
                          tf.multiply(lambda_noobj, yolo_loss[4])])
    yolo_loss = tf.reduce_sum(yolo_loss)

    return yolo_loss


def l_relu(features):
    return tf.nn.leaky_relu(features, 0.1)


# Below code need optimization may be by using Variable Scope.
def squeeze_module(x, dim, idx):
    name = 'conv_' + idx + '_sq'
    return tf.layers.conv2d(x, filters=dim, kernel_size=1, strides=1, padding="same",
                            activation=l_relu, name=name)


def expand_module(x, dim, idx):
    name = 'conv_' + idx + '_ex_' + '0'
    net1 = tf.layers.conv2d(x, filters=dim, kernel_size=1, strides=1, padding="same",
                            activation=l_relu, name=name)
    name = 'conv_' + idx + '_ex_' + '1'
    net2 = tf.layers.conv2d(x, filters=dim, kernel_size=3, strides=1, padding="same",
                            activation=l_relu, name=name)
    return tf.concat([net1, net2], 3)


def fire_module(input_tensor, squeeze_dim, expand_dim, idx):
    net = squeeze_module(input_tensor, squeeze_dim, idx)
    net = expand_module(net, expand_dim, idx)
    return net
