import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# import einops

from .layers import HilbertLayer


class Filter1D(nn.Module):
    def __init__(self, n_channels, kernel_size, srate, fmin_init, fmax_init, freq=None, bandwidth=None, padding_mode='zeros', seed=None, hilbert=False):
        super().__init__()
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.srate = srate
        
        padding = kernel_size // 2
        if hilbert:
            self.padding_hilbert = kernel_size // 2
            padding += self.padding_hilbert
            
        if padding_mode == 'zeros':
            self.pad = nn.ConstantPad1d(padding, 0)
        elif padding_mode == 'reflect':
            self.pad = nn.ReflectionPad1d(padding)
        
        self.register_buffer('_scale', torch.arange(-self.kernel_size//2 + 1, self.kernel_size//2 + 1).reshape((1,1,-1)) / self.srate)

        if seed is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        self.freq = freq
        if self.freq is None:
            coef_freq = self._create_parameters_freq(self.n_channels, fmin_init, fmax_init)
            self.coef_freq = nn.Parameter(coef_freq)
        else:
            self.register_buffer('_freq', freq)
        
        self.bandwidth = bandwidth
        if self.bandwidth is None:
            coef_bandwidth = self._create_parameters_bandwidth(self.n_channels)
            self.coef_bandwidth = nn.Parameter(coef_bandwidth)
        else:
            if not isinstance(bandwidth, torch.Tensor):
                bandwidth = torch.tensor(bandwidth, dtype=torch.float32).reshape((1,))
            assert bandwidth.shape[0] in (1, self.n_channels)
            if bandwidth.shape[0] != self.n_channels:
                bandwidth = bandwidth.repeat(self.n_channels)
            self.register_buffer('_bandwidth', bandwidth)
        
    def _create_parameters_freq(self, n_coef, fmin_init, fmax_init):
        coef = fmin_init + torch.rand(size=(n_coef,)) * (fmax_init - fmin_init)
        return coef
    
    def _create_parameters_bandwidth(self, n_coef):
        coef = torch.rand(size=(n_coef,)) * 0.95 + 0.025
        coef = torch.log(coef / (1-coef))
        return coef
    
    def _create_frequencies(self):
        if self.freq is None:
            freq = 0.2 + F.softplus(self.coef_freq)
        else:
            freq = self._freq
            
        if self.bandwidth is None:
            bandwidth = torch.sigmoid(self.coef_bandwidth) * 0.95 + 0.025
        else:
            bandwidth = self._bandwidth
        bandwidth = bandwidth * freq
        
        freq_low = freq - bandwidth / 2
        freq_high = freq + bandwidth / 2

        return freq, bandwidth, freq_low, freq_high
    
    
class SincLayer(Filter1D):
    def __init__(self, n_channels, kernel_size, srate, fmin_init, fmax_init, freq=None, bandwidth=None, padding_mode='zeros', seed=None):
        super().__init__(n_channels, kernel_size, srate, fmin_init, fmax_init, freq, bandwidth, padding_mode, seed)
        
        self.register_buffer('_hamming_window', torch.hamming_window(kernel_size).reshape((1,1,-1)))
#         self.hilbert = HilbertLayer()

    def _create_filters(self, freq_low, freq_high):
        freq_low, freq_high = freq_low.reshape((-1,1,1)), freq_high.reshape((-1,1,1))   
        filt_low = freq_low * torch.special.sinc(2 * freq_low * self._scale)
        filt_high = freq_high * torch.special.sinc(2 * freq_high * self._scale)
        filt = self._hamming_window * 2 * (filt_high - filt_low) / self.srate
        return filt
        
    def forward(self, x):
        x = self.pad(x)
        _, _, freq_low, freq_high = self._create_frequencies()
        filt = self._create_filters(freq_low, freq_high)
        x = F.conv1d(x, filt, groups=x.shape[-2], padding='valid')
        return x
    
    
class SincHilbertLayer(Filter1D):
    def __init__(self, n_channels, kernel_size, srate, fmin_init, fmax_init, freq=None, bandwidth=None, padding_mode='zeros', seed=None):
        super().__init__(n_channels, kernel_size, srate, fmin_init, fmax_init, freq, bandwidth, padding_mode, seed, hilbert=True)
        
        self.register_buffer('_hamming_window', torch.hamming_window(kernel_size).reshape((1,1,-1)))
        self.hilbert = HilbertLayer()

    def _create_filters(self, freq_low, freq_high):
        freq_low, freq_high = freq_low.reshape((-1,1,1)), freq_high.reshape((-1,1,1))   
        filt_low = freq_low * torch.special.sinc(2 * freq_low * self._scale)
        filt_high = freq_high * torch.special.sinc(2 * freq_high * self._scale)
        filt = self._hamming_window * 2 * (filt_high - filt_low) / self.srate
        return filt
        
    def forward(self, x, return_filtered=False):
        x = self.pad(x)
        _, _, freq_low, freq_high = self._create_frequencies()
        filt = self._create_filters(freq_low, freq_high)
        x = F.conv1d(x, filt, groups=x.shape[-2], padding='valid')
            
        if not return_filtered:
            x = self.hilbert(x)
            x = torch.abs(x)
        x = x[...,self.padding_hilbert:-self.padding_hilbert]
        return x
    


class WaveletLayer(Filter1D):
    def __init__(self, n_channels, kernel_size, srate, fmin_init, fmax_init, freq=None, bandwidth=None, padding_mode='zeros', seed=None):
        super().__init__(n_channels, kernel_size, srate, fmin_init, fmax_init, freq, bandwidth, padding_mode, seed)
           
    def _create_filters(self, freq, bandwidth):
        freq, bandwidth = freq.reshape((-1,1,1)), bandwidth.reshape((-1,1,1))
        sigma2 = (2 * math.log(2)) / (bandwidth * math.pi)**2
        filt = (2 * math.pi * sigma2)**(-1/2) / (self.srate / 2)
        filt = filt * torch.cos(2*math.pi * freq * self._scale)
        filt = filt * torch.exp(- self._scale**2 / (2 * sigma2))
        return filt
                            
    def forward(self, x):
        x = self.pad(x)
        freq, bandwidth, _, _ = self._create_frequencies()
        filt = self._create_filters(freq, bandwidth)
        x = F.conv1d(x, filt, groups=x.shape[-2], padding='valid')
        return x

        
        
class ComplexWaveletLayer(Filter1D):
    def __init__(self, n_channels, kernel_size, srate, fmin_init, fmax_init, freq=None, bandwidth=None, padding_mode='zeros', seed=None):
        super().__init__(n_channels, kernel_size, srate, fmin_init, fmax_init, freq, bandwidth, padding_mode, seed)
           
    def _create_filters(self, freq, bandwidth):
        freq, bandwidth = freq.reshape((-1,1,1)), bandwidth.reshape((-1,1,1))
        sigma2 = (2 * math.log(2)) / (bandwidth * math.pi)**2
        filt = (2 * math.pi * sigma2)**(-1/2) / (self.srate / 2)
        filt = filt * (torch.exp(1j*2*math.pi * freq * self._scale) - torch.exp(-0.5*(2*math.pi * freq)**2))
        filt = filt * torch.exp(- self._scale**2 / (2 * sigma2))
        return filt
          
    def forward(self, x, return_filtered=False):
        x = self.pad(x)
        freq, bandwidth, _, _ = self._create_frequencies()
        filt = self._create_filters(freq, bandwidth)
        
        if return_filtered:
            x = F.conv1d(x, filt.real, groups=x.shape[-2], padding='valid')
        else:
            x = x.to(torch.complex64)
            x = F.conv1d(x, filt, groups=x.shape[-2], padding='valid')
            x = torch.abs(x)
        return x