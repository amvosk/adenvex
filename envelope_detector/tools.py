import torch
from torch.autograd import grad
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import einops
import scipy.signal as sg

from .utils import (
    SimpleDataset,
    covariance,
    spectrum,
    check_spatial_filters,
    check_temporal_filters,
)

def create_importance_indices(
    model, data, order=1, nbatch=100, device="cpu", grad_out=False
):
    """
    Computes the importance indices of the model's output with respect
    to its intermediate outputs (`z`) based on gradient values.

    The function passes the input data through the model,
    computes gradients of the model's output (`y`) with respect to its
    intermediate outputs (`z`), and then aggregates these gradients
    to determine the importance of each element in `z`.

    Parameters:
    ----------
    model : torch.nn.Module
        The PyTorch model for which the importance indices need to be computed.
        The model is expected to return intermediate outputs `z`
        and final outputs `y` when called.

    data : torch.Tensor or array-like
        Input data to be passed through the model.
        The shape is expected to be compatible with the model's input shape.

    order : int, optional (default=1)
        Order of the norm for gradient aggregation.

    nbatch : int, optional (default=100)
        Batch size for processing the data in chunks.

    device : str, optional (default='cpu')
        The device to which the model and data will be moved before computation.
        Typically 'cuda' for GPU and 'cpu' for CPU.

    Returns:
    -------
    importance_indices : torch.Tensor
        Indices sorted by importance based on the aggregated gradient values.
        The index at the 0th position represents
        the most important element in `z` and so on.

    gradients : torch.Tensor
        The aggregated gradients for all elements in `z`.
    """
    if isinstance(data, Dataset):
        dataset = data
    elif (len(data.shape) == 3) and (data.shape[0] != 1):
        dataset = SimpleDataset(data)
        
    dataloader = DataLoader(dataset, batch_size=nbatch, shuffle=False)
    gradients = 0
    model = model.to(device)

    for batch in dataloader:
        if isinstance(batch, tuple) or isinstance(batch, list):
            batch = (x.to(device) for x in batch)
        else:
            batch = batch.to(device)
        z, y = model(batch)
        # jacobian-vector product used to determine feature importance
        gradients_ = grad(
            outputs=y, inputs=z, grad_outputs=torch.ones_like(y), retain_graph=True
        )[0]
        gradients_norm = torch.sum(torch.abs(gradients_) ** order, dim=(0, -1)) ** (
            1 / order
        )
        gradients += gradients_norm

    importance_indices = torch.sort(gradients, descending=True).indices

    return importance_indices, gradients


def create_spatial_patterns(
    x, spatial_filters, temporal_filters, nbatch=100, device="cpu", dtype=torch.float32
):
    """
    Computes the spatial patterns for a given dataset
    using provided spatial and temporal filters.

    Parameters:
    ----------
    x : torch.Tensor
        The input dataset. It can be 2D (single sample) or 3D (batched samples).

    spatial_filters : torch.Tensor
        Spatial filters used for analysis.
        Shape: [number of output channels, number of input channels]

    temporal_filters : torch.Tensor
        Temporal filters used for temporal unmixing.
        Shape: [number of output channels, size of temporal filter]

    nbatch : int, optional (default=100)
        Batch size for processing the data if the input is batched.

    device : str, optional (default="cpu")
        Torch device, either "cpu" or "cuda".

    dtype : torch.dtype, optional (default=torch.float32)
        Desired data type for the tensors.

    Returns:
    -------
    results : dict
        Contains:
            - 'spatial_filters': The spatial filters used.
            - 'spatial_patterns': The derived spatial patterns.
    """
    
    if isinstance(x, Dataset):
        batched = True
        dataloader = DataLoader(dataset, batch_size=nbatch, shuffle=False)
    elif (len(x.shape) == 2) or (len(x.shape) == 3 and x.shape[0] == 1):
        batched = False
        if type(x) is not torch.Tensor:
            x = torch.tensor(x, dtype=dtype)
        x = x.to(device)
    elif (len(x.shape) == 3) and (x.shape[0] != 1):
        batched = True
        dataset = SimpleDataset(x, dtype)
        dataloader = DataLoader(dataset, batch_size=nbatch, shuffle=False)
    else:
        raise ValueError(
            f"""
            Input with shape {x.shape} is not supported, 
            len(input.shape) has to be in (2, 3).
            """
        )

    check_spatial_filters(spatial_filters)
    check_temporal_filters(temporal_filters)
    assert (
        x.shape[-2] == spatial_filters.shape[1]
    ), "number of input channels of data and spatial filters should be equal"
    assert (
        spatial_filters.shape[0] == temporal_filters.shape[0]
    ), "number of output channels of spatial and temporal filters should be equal"
    nfeatures, nchannels = spatial_filters.shape

    spatial_filters = spatial_filters.to(dtype=dtype).to(device)
    temporal_filters = temporal_filters.to(dtype=dtype).to(device)

    spatial_patterns = torch.zeros((nfeatures, nchannels), dtype=dtype, device=device)
    for projection in range(nfeatures):
        temporal_filter = temporal_filters[projection]
        spatial_filter = spatial_filters[projection]
        temporal_filter_ = einops.repeat(temporal_filter, "t -> c 1 t", c=nchannels)

        if not batched:
            x_filtered = F.conv1d(
                x,
                temporal_filter_,
                bias=None,
                padding="same",
                groups=nchannels,
            )
            x_cov = covariance(x_filtered, unbiased=True)
        else:
            x_mean = 0
            for batch in dataloader:
                batch = batch.to(device)
                batch_size = batch.shape[0]
                batch = einops.rearrange(batch, "b c t -> c (b t)")
                batch_filtered = F.conv1d(
                    batch,
                    temporal_filter_,
                    bias=None,
                    padding="same",
                    groups=nchannels,
                )
                batch_filtered = einops.rearrange(
                    batch_filtered, "c (b t) -> b c t", b=batch_size
                )
                x_mean += (
                    torch.sum(batch_filtered, dim=(0, -1), keepdims=True)
                    / batch.shape[-1]
                )
            x_mean /= len(dataset)

            x_cov = 0
            for batch in dataloader:
                batch = batch.to(device)
                batch_size = batch.shape[0]
                batch = einops.rearrange(batch, "b c t -> c (b t)")
                batch_filtered = F.conv1d(
                    batch,
                    temporal_filter_,
                    bias=None,
                    padding="same",
                    groups=nchannels,
                )
                batch_filtered = einops.rearrange(
                    batch_filtered, "c (b t) -> b c t", b=batch_size
                )
                x_cov_batch = covariance(batch_filtered, unbiased=False, mean=x_mean)
                x_cov += torch.sum(x_cov_batch, dim=0) * batch.shape[-1]
            x_cov /= batch.shape[-1] * len(dataset) - 1

        #         print(x_cov.shape, )
        pattern = torch.einsum("...cC, C -> c", x_cov, spatial_filter)
        spatial_patterns[projection] = pattern

    results = {
        "spatial_filters": spatial_filters,
        "spatial_patterns": spatial_patterns,
    }
    return results


def create_temporal_patterns(
    x,
    spatial_filters,
    temporal_filters,
    fs=1000,
    nyquist=500,
    nbatch=100,
    device="cpu",
    dtype=torch.float32,
):
    """
    Computes the temporal patterns of a given dataset
    using provided spatial and temporal filters.

    Parameters:
    ----------
    x : torch.Tensor
        The input dataset. It can be 2D (single sample) or 3D (batched samples).

    spatial_filters : torch.Tensor
        Spatial filters used for spatial unmixing.
        Shape: [number of output channels, number of input channels]

    temporal_filters : torch.Tensor
        Temporal filters used for analysis.
        Shape: [number of output channels, size of temporal filter]

    fs : int, optional (default=1000)
        Sampling frequency of the input data.

    nyquist : int, optional (default=500)
        Nyquist rate, typically half of the sampling frequency.

    nbatch : int, optional (default=100)
        Batch size for processing the data if the input is batched.

    device : str, optional (default="cpu")
        Torch device, either "cpu" or "cuda".

    dtype : torch.dtype, optional (default=torch.float32)
        Desired data type for the tensors.

    Returns:
    -------
    results : dict
        Contains:
            - 'frequencies': Frequencies over which PSD is presented.
            - 'input_spectrum': The PSD of the input.
            - 'temporal_filters_spectrum': The spectrum tempotal filters.
            - 'temporal_patterns_spectrum': The spectrum tempotal pattern.
            - 'output_spectrum': The PSD of the output.
    """
    if (len(x.shape) == 2) or (len(x.shape) == 3 and x.shape[0] == 1):
        batched = False
        if type(x) is not torch.Tensor:
            x = torch.tensor(x, dtype=dtype)
        x = x.to(device)
    elif (len(x.shape) == 3) and (x.shape[0] != 1):
        batched = True
        dataset = SimpleDataset(x, dtype)
        dataloader = DataLoader(dataset, batch_size=nbatch, shuffle=False)
    else:
        raise ValueError(
            f"""
            Input with shape {x.shape} is not supported, 
            len(input.shape) has to be in (2, 3).
            """
        )

    check_spatial_filters(spatial_filters)
    check_temporal_filters(temporal_filters)
    assert (
        x.shape[-2] == spatial_filters.shape[1]
    ), "number of input channels of data and spatial filters should be equal"
    assert (
        spatial_filters.shape[0] == temporal_filters.shape[0]
    ), "number of output channels of spatial and temporal filters should be equal"
    nfeatures, nchannels = spatial_filters.shape

    spatial_filters = spatial_filters.to(dtype=dtype).to(device)
    temporal_filters = temporal_filters.to(dtype=dtype).to(device)

    if not batched:
        if len(x.shape) != 2:
            x = einops.rearrange(x, "1 i t -> i t")
        x_unmixed = torch.einsum("oi, it -> ot", spatial_filters, x)

        x_unmixed_numpy = x_unmixed.cpu().detach().numpy()
        input_frequencies, input_spectrum = sg.welch(
            x_unmixed_numpy, fs=fs, nperseg=nyquist, detrend="constant", axis=-1
        )
        input_spectrum = torch.tensor(
            input_spectrum[..., :-1], dtype=dtype, device=device
        )
    else:
        input_spectrum = 0
        for batch in dataloader:
            batch_size = batch.shape[0]
            batch = batch.to(device)
            batch_unmixed = torch.einsum("oi, bit -> bot", spatial_filters, batch)
            batch_unmixed = einops.rearrange(batch_unmixed, "b o t -> o (b t)")
            batch_unmixed_numpy = batch_unmixed.cpu().detach().numpy()
            input_frequencies, input_spectrum_ = sg.welch(
                batch_unmixed_numpy,
                fs=fs,
                nperseg=nyquist,
                detrend="constant",
                axis=-1,
            )
            input_spectrum_ = torch.tensor(
                input_spectrum_[..., :-1],
                dtype=dtype,
                device=device,
            )
            input_spectrum += input_spectrum_ * batch_size
        input_spectrum /= len(dataset)

    input_frequencies = torch.tensor(
        input_frequencies[:-1],
        dtype=dtype,
        device=device,
    )
    frequencies_filter, temporal_filters_spectrum = spectrum(
        temporal_filters, fs, nfreq=nyquist
    )

    results = {
        "frequencies": input_frequencies,
        "input_spectrum": input_spectrum,
        "temporal_filters_spectrum": temporal_filters_spectrum,
        "temporal_patterns_spectrum": temporal_filters_spectrum * input_spectrum,
        "output_spectrum": torch.pow(temporal_filters_spectrum, 2) * input_spectrum,
    }
    return results
