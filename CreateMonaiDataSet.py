from torch.utils.data import Dataset
import dicom


class CreateMonaiDataset(Dataset):
    def __init__(self, df, labels, transform):
        self.df = df.reset_index()
        self.labels = labels
        self.transform = transform

    # This function just returns the length of the labels when called
    def __len__(self):
        return self.df.shape[0]

    '''
        This function is used by Pytorch’s Dataset module to get a sample and construct the dataset. When initialised, it will loop through this function creating a sample from each instance in the dataset.
        index parameter: passed in to the function is a number, this number is the data instance which Dataset will be looping through
    '''

    def __getitem__(self, index):
        ds_row = self.df.iloc[index]
        return dicom.load_train_samples_dicom(ds_row)
