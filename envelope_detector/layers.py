import torch
import torch.nn as nn
import einops

class HilbertLayer(nn.Module):
    """
    A PyTorch layer that computes the Hilbert transform of the input data.

    The Hilbert transform is used to obtain the analytic signal from a real-valued function. This layer
    effectively broadens the spectrum of the signal using the Hilbert transform, which can be useful
    for various signal processing applications.

    Methods:
    -------
    forward(x: torch.Tensor) -> torch.Tensor:
        Compute the Hilbert transform of the input tensor `x`.

    Parameters:
    ----------
    x : torch.Tensor
        The input tensor, typically a time-domain signal, to which the Hilbert transform is applied.
        It is expected that the last dimension of `x` is the time or sequence dimension.

    Returns:
    -------
    x_hilbert : torch.Tensor
        The analytic signal obtained by applying the Hilbert transform to `x`.

    Examples:
    --------
    >>> hilbert_layer = HilbertLayer()
    >>> signal = torch.randn(1, 128)  # Example data
    >>> transformed_signal = hilbert_layer(signal)

    Notes:
    -----
    The Hilbert transform is applied in the frequency domain using the FFT (Fast Fourier Transform)
    and its inverse IFFT for efficient computation.
    """

    def __init__(self):
        super(self.__class__, self).__init__()

    def forward(self, x):
        N = x.shape[-1]
        xf = torch.fft.fft(x, dim=-1)

        h = torch.zeros(xf.shape, dtype=x.dtype, device=x.device)
        h[..., 0] = h[..., N // 2] = 1
        h[..., 1 : (N + 1) // 2] = 2

        x_hilbert = torch.fft.ifft(xf * h, dim=-1)
        return x_hilbert


class HilbertAmplitudeLayer(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.hilbert = HilbertLayer()

    def forward(self, x):
        x = self.hilbert(x)
        x = torch.abs(x)
        return x
    
class HilbertSplitLayer(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.hilbert = HilbertLayer()

    def forward(self, x):
        x_complex = self.hilbert(x)
        x_real = torch.view_as_real(x_complex)
        x = einops.rearrange(x_real, 'b c t a -> b (c a) t')
        return x
