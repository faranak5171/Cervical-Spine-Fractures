# %%
import os
import sys
import gc
import ast
import cv2
import time
import timm
from timm.models.layers.conv2d_same import Conv2dSame
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
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold, StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from monai.transforms import Resize
import monai.transforms as transforms

import SegmentationModel
import dicom 

import warnings
warnings.filterwarnings("ignore")

# %%
"""
**Define required directories**
"""

# %%

data_dir = 'D:\RSNA-2022-cervical-spine-fracture-detection'
log_dir = './logs'
model_dir = './models'

os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# %%
"""
**Define Transformations for input images to monai's inputs**
"""

# %%
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

transform_valid_data = transforms.Compose()

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
"""
**Load DataFrames**
"""

# %%
df_train = pd.read_csv(os.path.join(data_dir,'train.csv'))
df_train.head()

# %%
mask_files = os.listdir(f"{data_dir}/segmentations")
print(f"Number of mask files: {len(mask_files)}")

df_mask = pd.DataFrame({"mask_path":mask_files})
df_mask["StudyInstanceUID"] = df_mask["mask_path"].apply(lambda x:x[:-4])
df_mask["mask_path"] = df_mask["mask_path"].apply(lambda x: os.path.join(data_dir,'segmentations',x))
df_mask.head()

# %%
df = df_train.merge(df_mask, on='StudyInstanceUID', how='left')
df['image_folder'] = df['StudyInstanceUID'].apply(lambda x: os.path.join(data_dir,'train_images',x))
df['mask_path'].fillna('',inplace=True)
print(df.shape)
df.head()

# %%
df_segments = df.query('mask_path != ""').reset_index(drop=True)
print(df_segments.shape)
df_segments.head()

# %%
"""
**K-Fold Cross Validation**
"""

# %%
# Split data into train and validation set with KFOLD Cross validation
k = 5
kf = KFold(n_splits=k, random_state=None)

df_segments['fold'] = -1
for fold, (train_idx, valid_idx) in enumerate(kf.split(df_segments)):
    df_segments.loc[valid_idx,'fold'] = fold

df_segments.head()

# %%
"""
**Convert 2D input to 3D** 

DICOM images are in axial format, while NIFTI files are in Sagittal plane. So, We should convert the segmentation model to 3D model.
"""

# %%
'''
    3D transformers  to change 2d inputs (DICOM Images)to 3d outputs (NIFTI Images)
'''
def Convert_2DT3D(module):
    module_output = module
    if isinstance(module, torch.nn.BatchNorm2d):
        module_output = torch.nn.BatchNorm3d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    elif isinstance(module, Conv2dSame):
        print("conv2Dsame")
    elif isinstance(module, torch.nn.Conv2d):
        module_output = torch.nn.Conv3d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride= module.stride[0],
            padding= module.padding[0],
            dilation=module.dilation[0],
            bias = module.bias is not None,
            padding_mode= module.padding_mode
        )
        module_output.weight = torch.nn.Parameter(module.weight.unsqueeze(-1).repeat(1,1,1,1,module.kernel_size[0]))

    elif isinstance(module, torch.nn.MaxPool2d):
        module_output = torch.nn.MaxPool3d(
            kernel_size= module.kernel_size,
            stride= module.stride,
            padding= module.padding,
            dilation=module.dilation,
            ceil_mode=module.ceil_mode
        )
    elif isinstance(module, torch.nn.AvgPool2d):
        module_output = torch.nn.AvgPool3d(
            kernel_size= module.kernel_size,
            stride=module.stride,
            padding= module.padding,
            ceil_mode= module.ceil_mode
        )
    for name, child in module.named_children():
        module_output.add_module(name, Convert_2DT3D(child))
    return module_output

# %%
for fold in range(5):
    ds_train = df_segments[df_segments['fold'] != fold].reset_index(drop=True)
    ds_valid_ = df_segments[df_segments['fold'] == fold].reset_index(drop=True)

    # dataset_valid = CreateMonaiDataset(ds_valid_, 'valid', transform=transform_valid_data)    
    # loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=4)    
    # loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=4, shuffle=False, num_workers=4)
    # model = SegmentationModel.SegModel('resnet18')
    # model = Convert_2DT3D(model)
    # model = model.to(device)