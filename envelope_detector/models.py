import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio.transforms

import einops
import scipy.signal as sg

from .utils import SimpleDataset


class EnvelopeDetector(nn.Module):
    """
    An interpretable envelope detector for decoding and interpreting cortical signals.

    The class represents a PyTorch model that applies various filters, 
    activation methods, and downsampling techniques to input data.

    Parameters:
    ----------
    nchannels : int
        Number of input channels.

    nfeatures : int
        Number of features after spatial filtering.

    temporal_filter_size : int, optional (default=7)
        Kernel size for the temporal convolution layer.

    downsample_coef : int, optional (default=1)
        Ratio between the input and output sampling rates.

    dropout : float, optional (default=0)
        Dropout rate after the temporal convolution layer.
        It's useful when the `temporal_filter_size` is large.

    activation_method : str, optional (default='demodulation')
        Specifies the activation method. Options include 'demodulation' which uses the absolute value,
        and 'hilbert' which employs the absolute value of the Hilbert transform.

    downsample_method : str, optional (default='avepool')
        Specifies the downsampling method. Options are:
        - 'avepool' which uses `nn.AvgPool1d`,
        - 'resample' which employs `torchaudio.transforms.Resample`,
        - 'none' which skips downsampling.

    fs_in : int, optional (default=1000)
        Sampling rate of the input data.

    use_temporal_smoother : bool, optional (default=False)
        Whether or not to use an additional temporal convolution layer after the activation layer.

    temporal_smoother_size : int, optional (default=3)
        Kernel size for the additional temporal convolution layer
        (only relevant if `use_temporal_smoother` is True).
    """

    def __init__(
        self,
        nchannels,
        nfeatures,
        temporal_filter_size=7,
        downsample_coef=1,
        dropout=0,
        activation_method="demodulation",
        downsample_method="avepool",
        fs_in=1000,
        use_temporal_smoother=False,
        temporal_smoother_size=3,
    ):
        super(EnvelopeDetector, self).__init__()

        # Properties
        self.nchannels = nchannels
        self.nfeatures = nfeatures
        self.temporal_filter_size = temporal_filter_size
        self.fs_in = fs_in
        self.downsample_coef = downsample_coef

        # Spatial filtering
        self.spatial_filter = nn.Conv1d(nchannels, nfeatures, kernel_size=1, bias=False)
        self.spatial_filter_batchnorm = nn.BatchNorm1d(nfeatures, affine=False)

        # Temporal filtering
        self.temporal_filter = nn.Conv1d(
            nfeatures,
            nfeatures,
            kernel_size=temporal_filter_size,
            bias=False,
            groups=nfeatures,
            padding="same",
        )
        self.temporal_filter_batchnorm = nn.BatchNorm1d(nfeatures, affine=False)

        # Dropout layer
        self.dropout_layer = nn.Dropout(p=dropout) if dropout else None

        # Activation
        if activation_method == "demodulation":
            self.activation = nn.LeakyReLU(-1)
        elif activation_method == "hilbert":
            self.activation = nn.Sequential(
                # Assuming HilbertLayer is defined elsewhere in your code
                HilbertLayer(),
                nn.LeakyReLU(-1),
            )

        # Temporal smoother
        self.temporal_smoother = (
            nn.Conv1d(
                nfeatures,
                nfeatures,
                kernel_size=temporal_smoother_size,
                groups=nfeatures,
                padding="same",
            )
            if use_temporal_smoother
            else None
        )

        # Downsampler
        if downsample_coef > 1:
            if downsample_method == "avepool":
                self.downsampler = nn.AvgPool1d(
                    kernel_size=downsample_coef, stride=downsample_coef
                )
            elif downsample_method == "resample":
                self.downsampler = torchaudio.transforms.Resample(
                    fs_in, fs_in / downsample_coef
                )
            elif downsample_method == "none":
                self.downsampler = None
        else:
            self.downsampler = None

    def forward(self, x):
        # If input is 2D, add a singleton batch dimension
        if len(x.shape) == 2:
            x = einops.rearrange(x, "C L -> 1 C L")

        # Spatial filtering
        x = self.spatial_filter(x)
        x = self.spatial_filter_batchnorm(x)

        # Temporal filtering
        x = self.temporal_filter(x)
        if self.dropout_layer:
            x = self.dropout_layer(x)
        x = self.temporal_filter_batchnorm(x)

        # Activation
        x = self.activation(x)

        # Temporal smoothing
        if self.temporal_smoother:
            x = self.temporal_smoother(x)

        # Downsampling
        if self.downsampler:
            x = self.downsampler(x)

        return x

    def _covariance(self, x, unbiased=True, mean=None):
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

    def spatial_patterns(self, x, nbatch=100):
        """
        Compute spatial patterns for a given dataset.
        """
        if (len(x.shape) == 2) or (len(x.shape) == 3 and x.shape[0] == 1):
            batched = False
            if type(x) is not torch.Tensor:
                x = torch.tensor(x, dtype=torch.float32)
            x = x.to(self.spatial_filter.weight.device)
        elif (len(x.shape) == 3) and (x.shape[0] != 1):
            batched = True
            dataset = SimpleDataset(x)
            dataloader = DataLoader(dataset, batch_size=nbatch, shuffle=False)
        else:
            raise ValueError(
                f"Input with shape {x.shape} is not supported, len(input.shape) has to be in (2, 3)."
            )

        spatial_patterns = torch.zeros(
            (self.nfeatures, self.nchannels),
            dtype=torch.float32,
            device=self.spatial_filter.weight.device,
        )
        for projection in range(self.nfeatures):
            temporal_filter = self.temporal_filter.weight.data[
                projection : projection + 1
            ]
            spatial_filter = self.spatial_filter.weight.data[projection]
            temporal_filter_ = einops.repeat(
                temporal_filter, "1 1 t -> c 1 t", c=self.nchannels
            )

            if not batched:
                x_filtered = F.conv1d(
                    x,
                    temporal_filter_,
                    bias=None,
                    padding="same",
                    groups=self.nchannels,
                )
                x_cov = self._covariance(x_filtered, unbiased=True)
            else:
                x_mean = 0
                for batch in dataloader:
                    batch = batch.to(self.temporal_filter.weight.device)
                    batch_size = batch.shape[0]
                    batch = einops.rearrange(batch, "b c t -> c (b t)")
                    batch_filtered = F.conv1d(
                        batch,
                        temporal_filter_,
                        bias=None,
                        padding="same",
                        groups=self.nchannels,
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
                    batch = batch.to(self.temporal_filter.weight.device)
                    batch_size = batch.shape[0]
                    batch = einops.rearrange(batch, "b c t -> c (b t)")
                    batch_filtered = F.conv1d(
                        batch,
                        temporal_filter_,
                        bias=None,
                        padding="same",
                        groups=self.nchannels,
                    )
                    batch_filtered = einops.rearrange(
                        batch_filtered, "c (b t) -> b c t", b=batch_size
                    )
                    x_cov_batch = self._covariance(
                        batch_filtered, unbiased=False, mean=x_mean
                    )
                    x_cov += torch.sum(x_cov_batch, dim=0) * batch.shape[-1]
                x_cov /= batch.shape[-1] * len(dataset) - 1

            pattern = torch.einsum("...cC, Ck -> c", x_cov, spatial_filter)
            spatial_patterns[projection] = pattern

        results = {
            "spatial_filters": einops.rearrange(
                self.spatial_filter.weight.data, "o i 1 -> o i"
            ),
            "spatial_patterns": spatial_patterns,
        }
        return results

    def _spectrum(self, signal, fs, nfreq):
        """
        Calculate the amplitude spectrum of a signal using FFT.
        """
        amplitudes = torch.abs(torch.fft.fft(signal, nfreq, dim=-1))
        frequencies = torch.fft.fftfreq(nfreq, d=1 / fs)
        positive_freq = nfreq // 2
        return frequencies[:positive_freq], amplitudes[..., :positive_freq]

    def temporal_patterns(self, x, fs=1000, nyquist=500, nbatch=100):
        """
        Derive temporal patterns of a given dataset.
        """
        if (len(x.shape) == 2) or (len(x.shape) == 3 and x.shape[0] == 1):
            batched = False
            if type(x) is not torch.Tensor:
                x = torch.tensor(x, dtype=torch.float32)
            x = x.to(self.spatial_filter.weight.device)
        elif (len(x.shape) == 3) and (x.shape[0] != 1):
            batched = True
            dataset = SimpleDataset(x)
            dataloader = DataLoader(dataset, batch_size=nbatch, shuffle=False)
        else:
            raise ValueError(
                f"Input with shape {x.shape} is not supported, len(input.shape) has to be in (2, 3)."
            )

        if not batched:
            if len(x.shape) != 2:
                x = einops.rearrange(x, "1 c t -> c t")
            x_unmixed = self.spatial_filter(x)

            x_unmixed_numpy = x_unmixed.cpu().detach().numpy()
            frequencies_input, spectrum_input = sg.welch(
                x_unmixed_numpy, fs=fs, nperseg=nyquist, detrend="constant", axis=-1
            )
            spectrum_input = torch.tensor(
                spectrum_input[..., :-1], dtype=x_unmixed.dtype, device=x_unmixed.device
            )
        else:
            spectrum_input = 0
            for batch in dataloader:
                batch = batch.to(self.temporal_filter.weight.device)
                batch_unmixed = self.spatial_filter(batch)
                batch_size = batch.shape[0]
                batch_unmixed = einops.rearrange(batch_unmixed, "b c t -> c (b t)")
                batch_unmixed_numpy = batch_unmixed.cpu().detach().numpy()
                frequencies_input, spectrum_input_ = sg.welch(
                    batch_unmixed_numpy,
                    fs=fs,
                    nperseg=nyquist,
                    detrend="constant",
                    axis=-1,
                )
                spectrum_input_ = torch.tensor(
                    spectrum_input_[..., :-1],
                    dtype=batch_unmixed.dtype,
                    device=batch_unmixed.device,
                )
                spectrum_input += spectrum_input_ * batch_size
            spectrum_input /= len(dataset)

        temporal_filter = self.temporal_filter.weight.data
        frequencies_input = torch.tensor(
            frequencies_input[:-1],
            dtype=temporal_filter.dtype,
            device=temporal_filter.device,
        )
        frequencies_filter, temporal_filter_amplitudes = self._spectrum(
            temporal_filter, fs, nfreq=nyquist
        )
        temporal_filter_amplitudes = einops.rearrange(
            temporal_filter_amplitudes, "c 1 t -> c t"
        )

        results = {
            "frequencies": frequencies_input,
            "input_spectrum": spectrum_input,
            "temporal_filters_spectrum": temporal_filter_amplitudes,
            "temporal_patterns_spectrum": temporal_filter_amplitudes * spectrum_input,
            "output_spectrum": torch.pow(temporal_filter_amplitudes, 2)
            * spectrum_input,
        }
        return results
