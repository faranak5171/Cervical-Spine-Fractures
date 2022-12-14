import os
import sys
import gc
import ast
import cv2
import time
import pickle
import random
import pydicom
import argparse
import warnings
import numpy as np
import pandas as pd
from glob import glob
import nibabel
from PIL import Image
from tqdm import tqdm
import albumentations
from pylab import rcParams
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold

import torch
import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from monai.transforms import Resize
import monai.transforms as transforms

import SegmentationModel
import warnings
warnings.filterwarnings("ignore")


data_dir = 'D:\RSNA-2022-cervical-spine-fracture-detection'
log_dir = './logs'
model_dir = './models'

os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

img_size = [128,128,128]
# Resize the input image to spatial size
monai_img_size = Resize(img_size)
#translate_range, a sequence of positive floats, is used to generate the n shift parameters
translate_range = [int(x*y) for x,y in zip(img_size, (0.3, 0.3, 0.3))]


transform_train_data = transforms.Compose([
    transforms.RandFlipd(keys=["image","mask"], prob=0.5, spatial_axis=1),
    transforms.RandFlipd(["image","mask"], prob=0.5, spatial_axis=2),
    transforms.RandAffined(keys=["image","mask"], translate_range=translate_range, padding_mode='zeros', prob=0.7),
    transforms.RandGridDistortiond(keys=("image", "mask"), prob=0.5, distort_limit=(-0.01, 0.01), mode="nearest")
])

df_train = pd.read_csv(os.path.join(data_dir,'train.csv'))
mask_files = os.listdir(f"{data_dir}/segmentations")


df_mask = pd.DataFrame({"mask_path":mask_files})
df_mask["StudyInstanceUID"] = df_mask["mask_path"].apply(lambda x:x[:-4])
df_mask["mask_path"] = df_mask["mask_path"].apply(lambda x: os.path.join(data_dir,'segmentations',x))

df = df_train.merge(df_mask, on='StudyInstanceUID', how='left')
df['image_folder'] = df['StudyInstanceUID'].apply(lambda x: os.path.join(data_dir,'train_images',x))
df['mask_path'].fillna('',inplace=True)


df_segments = df.query('mask_path != ""').reset_index(drop=True)

# Split data into train and validation set with KFOLD Cross validation
k = 5
kf = KFold(n_splits=k, random_state=None)

df_segments['fold'] = -1
for fold, (train_idx, valid_idx) in enumerate(kf.split(df_segments)):
    df_segments.loc[valid_idx,'fold'] = fold


def convert_3d(module):
    print(type(module))
    module_output = module
    print(type(module))
    if isinstance(module, torch.nn.BatchNorm2d):
        print("BatchNorm2d")
    elif isinstance(module, torch.nn.Conv2d):
        print("Conv2d")
    elif isinstance(module, torch.nn.MaxPool2d):
        print("Max Pooling 2d")
    elif isinstance(module, torch.nn.AvgPool2d):
        print("Average pooling 2d")
    for name, child in module.named_children():
        module_output.add_module(name, convert_3d(child))
    return module_output

# define model
drop_rate = 0.
drop_path_rate = 0.
model = SegmentationModel.SegModel('resnet18',drop_rate,drop_path_rate)
model = convert_3d(model)