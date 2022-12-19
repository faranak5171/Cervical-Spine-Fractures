from torch.utils.data import Dataset


class CreateMonaiDataset(Dataset):
    def __init__(self, df, dataset_type, transform):
        self.df = df.reset_index()
        self.ds_type = dataset_type
        self.transform = transform
