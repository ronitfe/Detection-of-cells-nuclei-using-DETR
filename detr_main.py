# -*- coding: utf-8 -*-
"""working DETR WO AUG
detr with or without augmentation (by comment)
"""

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive

from AverageMeter import AverageMeter
from detr_utils import (get_train_transforms, get_valid_transforms, num_queries, num_classes,
                        null_class_coef, LR, EPOCHS)
from DETRModel import DETRModel
from NucleiDataset import NucleiDataset
!cd
~
drive.mount('/content/drive')
# %cd '/content/drive/My Drive/Colab Notebooks/kaggle_2018'


# !git clone https://github.com/facebookresearch/detr.git

import os
import numpy as np
import pandas as pd
from datetime import datetime
import time
import random
from tqdm.autonotebook import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from sklearn.model_selection import StratifiedKFold
import cv2
import sys

sys.path.append('./detr/')
from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from glob import glob
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

"""# Configuration

Basic configuration for this model
"""

seed_everything(seed)

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

train_images = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
test_images = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

print('resize train images... ')
sys.stdout.flush()

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + "/" + id_
    img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    train_images[n] = img

sizes_test = []
print('resize test images ... ')
sys.stdout.flush()

for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + "/" + id_
    img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    test_images[n] = img

print('Done!')

path_bboxes_csv = "bboxes.csv"
if not os.path.isfile(path_bboxes_csv):
    bboxes = pd.DataFrame(columns=["train_id", "mask_id", "rotby_90", "x", "y", "w", "h"])
    row_count = 1
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + "/" + id_
        for mask_id, mask_file in enumerate(next(os.walk(path + '/masks/'))[2]):
            mask_ = imread(path + '/masks/' + mask_file)
            for r in range(4):
                bboxes.loc[row_count] = store_bounding_boxes(np.rot90(mask_, r), id_, mask_id, r)
                row_count += 1
    bboxes.to_csv(path_bboxes_csv, index=False)
else:
    bboxes = pd.read_csv(path_bboxes_csv)

GRID_DIM = 12
GRID_PIX = IMG_WIDTH // GRID_DIM
BATCH_SIZE = 14

train_ids_df = pd.DataFrame(columns=["idx", "id_"])
cnt = 0
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    train_ids_df.loc[cnt] = {"idx": n, "id_": id_}
    cnt += 1

train_ids_df = train_ids_df.set_index(['idx'])

# bboxes['grid_row'] = bboxes['y']//GRID_PIX
# bboxes['grid_column'] = bboxes['x']//GRID_PIX

# bboxes['grid_center_x'] = bboxes['grid_column'] * GRID_PIX + GRID_PIX/2
# bboxes['grid_center_y'] = bboxes['grid_row'] * GRID_PIX + GRID_PIX/2

# bboxes['box_center_x'] = bboxes.x + bboxes['w']/2
# bboxes['box_center_y'] = bboxes.y + bboxes['h']/2

# bboxes['x'] = (bboxes.box_center_x  )/(IMG_WIDTH)
# bboxes['y'] = (bboxes.box_center_y )/(IMG_HEIGHT)

# bboxes['w'] = bboxes.w/(IMG_WIDTH)
# bboxes['h'] = bboxes.h/(IMG_WIDTH)

bboxes['confidence'] = 1

bboxes['box_area'] = bboxes.w * bboxes.h

bboxes

bboxes.y.max()

marking = bboxes
marking.rename(columns={'train_id': 'image_id', 'mask_id': 'source'}, inplace=True)

# bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
# for i, column in enumerate(['x', 'y', 'w', 'h']):
#     marking[column] = bboxs[:,i]
# marking.drop(columns=['bbox'], inplace=True)

# Creating Folds
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

df_folds = marking[['image_id']].copy()
df_folds.loc[:, 'bbox_count'] = 1
df_folds = df_folds.groupby('image_id').count()
df_folds.loc[:, 'source'] = marking[['image_id', 'source']].groupby('image_id').min()['source']
df_folds.loc[:, 'stratify_group'] = np.char.add(
    df_folds['source'].values.astype(str),
    df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
)
df_folds.loc[:, 'fold'] = 0

for fold_number, (train_index, val_index) in enumerate(
        skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
    df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number



'''
code taken from github repo detr , 'code present in engine.py'
'''

matcher = HungarianMatcher()

weight_dict = weight_dict = {'loss_ce': 1, 'loss_bbox': 1, 'loss_giou': 1}

losses = ['labels', 'boxes', 'cardinality']


def train_fn(data_loader, model, criterion, optimizer, device, scheduler, epoch):
    model.train()
    criterion.train()

    summary_loss = AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))

    for step, (images, targets, image_ids) in enumerate(tk0):
        # print(images[0])
        # print(type(images[0]))
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        output = model(images)

        loss_dict = criterion(output, targets)
        weight_dict = criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        optimizer.zero_grad()

        losses.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        summary_loss.update(losses.item(), BATCH_SIZE)
        tk0.set_postfix(loss=summary_loss.avg)

    return summary_loss


def eval_fn(data_loader, model, criterion, device):
    model.eval()
    criterion.eval()
    summary_loss = AverageMeter()

    with torch.no_grad():

        tk0 = tqdm(data_loader, total=len(data_loader))
        for step, (images, targets, image_ids) in enumerate(tk0):

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            output = model(images)

            loss_dict = criterion(output, targets)
            weight_dict = criterion.weight_dict

            losses = sum(
                loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            summary_loss.update(losses.item(), BATCH_SIZE)
            tk0.set_postfix(loss=summary_loss.avg)

    return summary_loss


def collate_fn(batch):
    return tuple(zip(*batch))


def run(fold):
    print(marking.head())
    df_train = df_folds[df_folds['fold'] != fold]
    df_valid = df_folds[df_folds['fold'] == fold]

    train_dataset = NucleiDataset(
        image_ids=df_train.index.values,
        dataframe=marking,
        transforms=get_train_transforms()
    )

    valid_dataset = NucleiDataset(
        image_ids=df_valid.index.values,
        dataframe=marking,
        transforms=get_valid_transforms()
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    device = torch.device('cuda')
    model = DETRModel(num_classes=num_classes, num_queries=num_queries)
    print(model)
    model = model.to(device)
    criterion = SetCriterion(num_classes - 1, matcher, weight_dict, eos_coef=null_class_coef,
                             losses=losses)
    criterion = criterion.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_loss = 10 ** 5
    for epoch in range(EPOCHS):
        train_loss = train_fn(train_data_loader, model, criterion, optimizer, device,
                              scheduler=None, epoch=epoch)
        valid_loss = eval_fn(valid_data_loader, model, criterion, device)

        print('|EPOCH {}| TRAIN_LOSS {}| VALID_LOSS {}|'.format(epoch + 1, train_loss.avg,
                                                                valid_loss.avg))

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            print('Best model found for Fold {} in Epoch {}........Saving Model'.format(fold,
                                                                                        epoch + 1))
            torch.save(model.state_dict(), f'detr_best_{fold}.pth')


run(fold=0)


def view_sample(df_valid, model, device):
    valid_dataset = NucleiDataset(image_ids=df_valid.index.values,
                                  dataframe=marking,
                                  transforms=get_valid_transforms()
                                  )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn)
    # print(type(iter(valid_data_loader)))
    # print(next(iter(valid_data_loader)))
    images, targets, image_ids = next(iter(valid_data_loader))
    _, h, w = images[0].shape  # for de normalizing images

    images = list(img.to(device) for img in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    boxes = targets[0]['boxes'].cpu().numpy()
    boxes = [np.array(box).astype(np.int32) for box in
             A.augmentations.bbox_utils.denormalize_bboxes(boxes, h, w)]
    sample = images[0].permute(1, 2, 0).cpu().numpy()

    model.eval()
    model.to(device)
    cpu_device = torch.device("cpu")

    with torch.no_grad():
        outputs = model(images)

    outputs = [{k: v.to(cpu_device) for k, v in outputs.items()}]

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # for box in boxes:
    #     cv2.rectangle(sample,
    #               (box[0], box[1]),
    #               (box[2]+box[0], box[3]+box[1]),
    #               (220, 0, 0), 1)

    oboxes = outputs[0]['pred_boxes'][0].detach().cpu().numpy()
    oboxes = [np.array(box).astype(np.int32) for box in
              A.augmentations.bbox_utils.denormalize_bboxes(oboxes, h, w)]
    prob = outputs[0]['pred_logits'][0].softmax(1).detach().cpu().numpy()[:, 0]

    for box, p in zip(oboxes, prob):

        if p > 0.5:
            color = (0, 0, 220)
            cv2.rectangle(sample,
                          (box[0], box[1]),
                          (box[2] + box[0], box[3] + box[1]),
                          color, 1)

    ax.set_axis_off()
    ax.imshow(sample)


model = DETRModel(num_classes=num_classes, num_queries=num_queries)
model.load_state_dict(torch.load("./detr_best_0.pth"))
view_sample(df_folds[df_folds['fold'] == 0], model=model, device=torch.device('cuda'))
