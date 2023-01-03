from torch.utils.data import Dataset
import dicom
import numpy as np
import torch


class CreateMonaiDataset(Dataset):
    def __init__(self, df, labels, transform):
        self.df = df.reset_index()
        self.labels = labels
        self.transform = transform

    '''
        This function is used by Pytorch’s Dataset module to get a sample and construct the dataset. When initialised, it will loop through this function creating a sample from each instance in the dataset.
        index parameter: passed in to the function is a number, this number is the data instance which Dataset will be looping through
    '''

    def __getitem__(self, index):
        ds_row = self.df.iloc[index]
        image, mask = dicom.load_dicom_nibable(ds_row)

        revert_list = [
            '1.2.826.0.1.3680043.1363',
            '1.2.826.0.1.3680043.20120',
            '1.2.826.0.1.3680043.2243',
            '1.2.826.0.1.3680043.24606',
            '1.2.826.0.1.3680043.32071']

        if ds_row.StudyInstanceUID in revert_list:
            mask = mask[:, :, :, ::-1]

        res = self.transform({'image': image, 'mask': mask})
        image = res['image'] / 255.
        mask = res['mask']
        mask = (mask > 127).astype(np.float32)

        image, mask = torch.tensor(image).float(), torch.tensor(mask).float()

        return image, mask

    def __len__(self):
        return self.df.shape[0]
