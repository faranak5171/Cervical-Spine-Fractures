import pydicom
import cv2
import glob
import os
import numpy as np
import nibabel as nib
from monai.transforms import Resize


def load_meta_info(path):
    return pydicom.dcmread(path)


def load_dicom(path, slice_num):
    path = os.path.join(path, f"{slice_num}.dcm")
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = cv2.resize(data, (128, 128), interpolation=cv2.INTER_LINEAR)
    return data


def load_dicoms_per_UID(UID_path):
    dicom_files = glob.glob(f"{UID_path}/*")
    slice_nums = []
    for path in dicom_files:
        slice_nums.append(path.split("\\")[-1].split('.')[0])
    images = []
    for slice_num in slice_nums:
        images.append(load_dicom(UID_path, slice_num))
    # Convert all images to numpy (num,width,height)
    images = np.array(images)
    # reverse shape (height, width,num)
    images = np.stack(images, -1)
    # Normalize image
    images = images - np.min(images)
    images = images / (np.max(images) + 1e-4)
    images = (images * 255).astype(np.uint8)
    return images


def load_dicom_nibable(row):
    image = load_dicoms_per_UID(row.image_folder)
    # add depth channel to image : shape (batch,height, width,num)
    image = np.expand_dims(image, axis=0).repeat(3, 0)

    # Load mask file with (Hight, width, channel)
    mask_file = nib.load(row.mask_path).get_fdata()
    # Pytorch requires images with CHW shape
    mask_file = mask_file.transpose(2, 0, 1)
    mask = np.zeros(
        shape=(7, mask_file.shape[1], mask_file.shape[2], mask_file.shape[0]))
    for cid in range(7):
        mask[cid] = (mask_file == (cid+1))
    mask = mask.astype(np.uint8)*255
    image_sizes = [128, 128, 128]
    R = Resize(image_sizes)
    mask = R(mask).ngitumpy()
    return image, mask
