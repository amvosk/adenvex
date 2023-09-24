import torch
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    """
    This class is used to convert an array-like structure into a PyTorch dataset,
    enabling compatibility with PyTorch's data utilities like DataLoader.
    The dataset is meant for single input data without any corresponding labels.

    Parameters:
    ----------
    data : array-like or torch.Tensor
        The input data to be wrapped into a dataset.
        This can be a list, numpy array, or a torch.Tensor.

    Methods:
    -------
    __len__() -> int:
        Returns the number of samples in the dataset.

    __getitem__(index: int) -> torch.FloatTensor:
        Returns a sample from the dataset at the specified index.
        If the original sample isn't a torch.FloatTensor,
        it's converted to one.
    """

    def __init__(self, data, dtype=torch.float32):
        self.data = data
        self.dtype = dtype

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        sample = (
            sample
            if type(sample) is torch.FloatTensor
            else torch.tensor(sample, dtype=self.dtype)
        )
        return sample


def covariance(x, unbiased=True, mean=None):
    """
    Compute the covariance matrix of a signal.
    """
    if mean is None:
        mean = torch.mean(x, dim=-1, keepdims=True)
    x = x - mean
    covariance_matrix = (
        1 / (x.shape[-1] - unbiased) * torch.einsum("...ct, ...Ct -> ...cC", x, x)
    )
    return covariance_matrix


def spectrum(x, fs, nfreq):
    """
    Calculate the amplitude spectrum of a signal using FFT.
    """
    amplitudes = torch.abs(torch.fft.fft(x, nfreq, dim=-1))
    frequencies = torch.fft.fftfreq(nfreq, d=1 / fs)
    positive_freq = nfreq // 2
    return frequencies[:positive_freq], amplitudes[..., :positive_freq]


def check_spatial_filters(spatial_filters):
    if len(spatial_filters.shape) != 2:
        raise ValueError(
            f"""
            Input with shape {spatial_filters.shape} for spatial filters is not supported, 
            len(input.shape) has to be equal to 2, where
            input.shape[0] is a number of output channels,
            input.shape[1] is a number of input channels.
            """
        )


def check_temporal_filters(temporal_filters):
    if len(temporal_filters.shape) != 2:
        raise ValueError(
            f"""
            Input with shape {temporal_filters.shape} for spatial filters is not supported, 
            len(input.shape) has to be equal to 2, where
            input.shape[0] is a number of output channels,
            input.shape[1] is a size of temporal filter.
            """
        )
