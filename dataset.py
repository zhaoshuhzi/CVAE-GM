import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class EEGDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.labels = df['label'].values
        self.data = df.drop(columns=['label']).values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), self.labels[idx]