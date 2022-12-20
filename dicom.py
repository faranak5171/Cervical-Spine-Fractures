import pydicom
import cv2
import glob
import os
import numpy as np


def load_dicom(path, slice_num):
    path = os.path.join(path,f"{slice_num}.dcm")
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = cv2.resize(data, (128,128),interpolation=cv2.INTER_LINEAR)
    return data


def load_dicoms_per_UID(UID_path):
    dicom_files = glob.glob(f"{UID_path}/*")
    slice_nums=[]
    for path in dicom_files:
        slice_nums.append(path.split("\\")[-1].split('.')[0])
    images = []
    for slice_num in slice_nums:
        images.append(load_dicom(UID_path, slice_num))
    # Convert all images to numpy (num,width,height)
    images = np.array(images)
    # reverse shape (width, height)
    images = np.stack(images, -1)
    #Normalize image
    images = images - np.min(images)
    images = images / (np.max(images) + 1e-4)
    images = (images * 255).astype(np.uint8)  
    return images


def load_train_samples_dicom(row):
    image = load_dicoms_per_UID(row.image_folder)
    # add depth channel to image shape
    image = np.expand_dims(image, axis=0).repeat(3,0)
    return image
