from torch.utils.data import Dataset
import dicom
import pandas as pd


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
        return image, mask

    def __len__(self):
        return self.df.shape[0]
