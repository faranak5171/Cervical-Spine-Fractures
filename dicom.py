import pydicom
import cv2
import glob
import os

def load_dicom(path, image_sizes):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data =cv2.resize(data, (image_sizes[0], image_sizes[0]), interpolation=cv2.INTER_LINEAR)
    return data

def load_dicom_line_par(path):
    t_paths = sorted(glob(os.path.join(path, "*")), key=lambda x:int(x.split('/')[-1].split('.')[0]))
    n_scans = len(t_paths)
    #indices = np.quantile(list(range(n_scans)), np.linspace())
