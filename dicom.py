import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import gdcm
import cv2
from glob import glob
import os
import numpy as np
import nibabel as nib
from monai.transforms import Resize

'''
    Because of the complexity in interpreting the pixel data.
    Pydicom provides an easy way to get it in a convenient form: Dataset.pixel_array.
'''


def load_dicom(path):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = cv2.resize(data, (128, 128), interpolation=cv2.INTER_LINEAR)
    return data


def load_dicom_line_par(path):

    t_paths = sorted(glob(os.path.join(path, "*")),
                     key=lambda x: int(x.split("\\")[-1].split(".")[0]))

    n_scans = len(t_paths)
    # take evenly spaced images to equal the image_size desired and resize those slices only rather than resizing the entire volume!
    indices = np.quantile(list(range(n_scans)), np.linspace(
        0., 1., 128)).round().astype(int)
    t_paths = [t_paths[i] for i in indices]

    images = []
    for filename in t_paths:
        images.append(load_dicom(filename))
    images = np.stack(images, -1)

    images = images - np.min(images)
    images = images / (np.max(images) + 1e-4)
    images = (images * 255).astype(np.uint8)

    return images


def load_dicom_nibable(row, has_mask=True):
    image = load_dicom_line_par(row.image_folder)
    # add depth channel to image : shape (batch,height, width,num)
    if image.ndim < 4:
        image = np.expand_dims(image, 0).repeat(3, 0)

    # Load mask files with nibabel library
    # get_fdata() means frame date , it returns the values of all dicom slices (Hight, width, channel)
    if has_mask:
        mask_org = nib.load(row.mask_path).get_fdata()
        shape = mask_org.shape
        mask_org = mask_org.transpose(1, 0, 2)[::-1, :, ::-1]  # (d, w, h)
        mask = np.zeros((7, shape[0], shape[1], shape[2]))
        for cid in range(7):
            mask[cid] = (mask_org == (cid+1))
        mask = mask.astype(np.uint8) * 255
        R = Resize([128, 128, 128])
        mask = R(mask).numpy()
        return image, mask
    else:
        return image


'''
    By default pydicom reads in pixel data as the raw bytes found in the file.
'''


def load_dicom_meta(path):
    return pydicom.dcmread(path)


'''
    The DICOM VOI LUT module applies a VOI or windowing operation to input values. 
    The apply_voi_lut() function can be used with an input array and a dataset containing a VOI LUT module to return values with applied VOI LUT or windowing. 
    When a dataset contains multiple VOI or windowing views then a particular view can be returned by using the index keyword parameter.
'''


def load_dicom_VOI(path):
    ds = pydicom.dcmread(path)
    img = apply_voi_lut(ds.pixel_array, ds)
    return img
