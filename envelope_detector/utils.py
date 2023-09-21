import torch
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    """
    A simple dataset wrapper for PyTorch that provides easy access and indexing to data.

    This class is used to convert an array-like structure into a PyTorch dataset, enabling compatibility with 
    PyTorch's data utilities like DataLoader. The dataset is meant for single input data without any corresponding labels.

    Parameters:
    ----------
    data : array-like or torch.Tensor
        The input data to be wrapped into a dataset. This can be a list, numpy array, or a torch.Tensor.

    Methods:
    -------
    __len__() -> int:
        Returns the number of samples in the dataset.

    __getitem__(index: int) -> torch.FloatTensor:
        Returns a sample from the dataset at the specified index. If the original sample isn't a torch.FloatTensor,
        it's converted to one.
    """
    
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        sample = sample if type(sample) is torch.FloatTensor else torch.tensor(sample, dtype=torch.float32)
        return sample