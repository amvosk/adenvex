import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# import einops

from .layers import HilbertLayer, TemporalPad


class TemporalFilter(nn.Module):
    def __init__(
        self,
        n_channels,
        kernel_size,
        srate,
        fmin_init=1,
        fmax_init=40,
        freq=None,
        bandwidth=None,
        margin_frequency=0.3,
        margin_bandwidth=0.05,
        seed=None,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.srate = srate
        self.margin_frequency = margin_frequency
        self.margin_bandwidth = margin_bandwidth
        
        self.register_buffer('_scale', torch.arange(-self.kernel_size//2 + 1, self.kernel_size//2 + 1) / self.srate)

        if seed is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        
        self.freq = freq
        if self.freq is None:
            coef_freq = self._create_parameters_freq(self.n_channels, fmin_init, fmax_init, seed)
            self.coef_freq = nn.Parameter(coef_freq)
        else:
            self.register_buffer('_freq', freq)
        
        self.bandwidth = bandwidth
        if self.bandwidth is None:
            coef_bandwidth = self._create_parameters_bandwidth(self.n_channels, seed)
            self.coef_bandwidth = nn.Parameter(coef_bandwidth)
        else:
            if not isinstance(bandwidth, torch.Tensor):
                bandwidth = torch.tensor(bandwidth, dtype=torch.float32).reshape((1,))
            assert bandwidth.shape[0] in (1, self.n_channels)
            if bandwidth.shape[0] != self.n_channels:
                bandwidth = bandwidth.repeat(self.n_channels)
            self.register_buffer('_bandwidth', bandwidth)
        
    def _create_parameters_freq(self, n_coef, fmin_init, fmax_init, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        coef = fmin_init + torch.rand(size=(n_coef,), generator=generator) * (fmax_init - fmin_init)
        return coef
    
    def _create_parameters_bandwidth(self, n_coef, seed):
        generator = torch.Generator()
        generator.manual_seed(seed+1)
        coef = torch.rand(size=(n_coef,), generator=generator) * 0.95 + 0.025
        coef = torch.log(coef / (1-coef))
        return coef
    
    def _create_frequencies(self):
        if self.freq is None:
            freq = self.margin_frequency + F.softplus(self.coef_freq)
        else:
            freq = self._freq

        if self.bandwidth is None:
            bandwidth = (torch.sigmoid(self.coef_bandwidth) * (1 - 2*self.margin_bandwidth) + self.margin_bandwidth)
        else:
            bandwidth = self._bandwidth
        bandwidth = bandwidth * freq
        
        freq_low = freq - bandwidth / 2
        freq_high = freq + bandwidth / 2

        return freq, bandwidth, freq_low, freq_high
    
    
    
    
class SincLayer1d(TemporalFilter):
    def __init__(self, in_channels, out_channels, kernel_size, srate, fmin_init, fmax_init, freq=None, bandwidth=None, padding_mode='zeros', seed=None):
        super().__init__(out_channels, kernel_size, srate, fmin_init, fmax_init, freq, bandwidth, seed=seed)
        self.in_channels = in_channels
        self.pad = TemporalPad(padding='same', dim='1d', kernel_size=kernel_size, padding_mode=padding_mode, hilbert=False)
        self.register_buffer('_hamming_window', torch.hamming_window(kernel_size).reshape((1,1,-1)))

    def _create_filters(self, freq_low, freq_high):
        _scale = self._scale.reshape((1,1,-1))
        freq_low, freq_high = freq_low.reshape((-1,1,1)), freq_high.reshape((-1,1,1))   
        filt_low = freq_low * torch.special.sinc(2 * freq_low * _scale)
        filt_high = freq_high * torch.special.sinc(2 * freq_high * _scale)
        filt = self._hamming_window * 2 * (filt_high - filt_low) / self.srate
        return filt
        
    def forward(self, x):
        x = self.pad(x)
        _, _, freq_low, freq_high = self._create_frequencies()
        filt = self._create_filters(freq_low, freq_high)
        assert self.in_channels == x.shape[-2]
        x = F.conv1d(x, filt, groups=self.in_channels, padding='valid')
        return x
    
                                     
class SincLayer2d(TemporalFilter):
    def __init__(self, in_channels, out_channels, kernel_size, srate, fmin_init, fmax_init, freq=None, bandwidth=None, padding_mode='zeros', seed=None):
        super().__init__(out_channels, kernel_size, srate, fmin_init, fmax_init, freq, bandwidth, seed=seed)
        self.in_channels = in_channels
        self.pad = TemporalPad(padding='same', dim='2d', kernel_size=kernel_size, padding_mode=padding_mode, hilbert=False)         
        self.register_buffer('_hamming_window', torch.hamming_window(kernel_size).reshape((1,1,1,-1)))
                                     
    def _create_filters(self, freq_low, freq_high):
        _scale = self._scale.reshape((1,1,1,-1))
        freq_low, freq_high = freq_low.reshape((-1,1,1,1)), freq_high.reshape((-1,1,1,1))   
        filt_low = freq_low * torch.special.sinc(2 * freq_low * _scale)
        filt_high = freq_high * torch.special.sinc(2 * freq_high * _scale)
        filt = self._hamming_window * 2 * (filt_high - filt_low) / self.srate
        return filt
        
    def forward(self, x):
        x = self.pad(x)
        _, _, freq_low, freq_high = self._create_frequencies()
        filt = self._create_filters(freq_low, freq_high)
        assert self.in_channels == x.shape[-3]
        x = F.conv2d(x, filt, groups=self.in_channels, padding='valid')
        return x
                                     
    
    
    
class SincHilbertLayer1d(TemporalFilter):
    def __init__(self, in_channels, out_channels, kernel_size, srate, fmin_init, fmax_init, freq=None, bandwidth=None, padding_mode='zeros', seed=None):
        super().__init__(out_channels, kernel_size, srate, fmin_init, fmax_init, freq, bandwidth, seed=seed)
        self.in_channels = in_channels
        self.pad = TemporalPad(padding='same', dim='1d', kernel_size=kernel_size, padding_mode=padding_mode, hilbert=True)   
        self.register_buffer('_hamming_window', torch.hamming_window(kernel_size).reshape((1,1,-1)))
        self.hilbert = HilbertLayer()

    def _create_filters(self, freq_low, freq_high):
        _scale = self._scale.reshape((1,1,-1))
        freq_low, freq_high = freq_low.reshape((-1,1,1)), freq_high.reshape((-1,1,1))   
        filt_low = freq_low * torch.special.sinc(2 * freq_low * _scale)
        filt_high = freq_high * torch.special.sinc(2 * freq_high * _scale)
        filt = self._hamming_window * 2 * (filt_high - filt_low) / self.srate
        return filt
        
    def forward(self, x, return_filtered=False):
        x = self.pad(x)
        _, _, freq_low, freq_high = self._create_frequencies()
        filt = self._create_filters(freq_low, freq_high)
        assert self.in_channels == x.shape[-2]
        x = F.conv1d(x, filt, groups=self.in_channels, padding='valid')
            
        if not return_filtered:
            x = self.hilbert(x)
            x = torch.abs(x)
        x = x[...,self.pad.padding_hilbert:-self.pad.padding_hilbert]
        return x
    
    
class SincHilbertLayer2d(TemporalFilter):
    def __init__(self, in_channels, out_channels, kernel_size, srate, fmin_init, fmax_init, freq=None, bandwidth=None, padding_mode='zeros', seed=None):
        super().__init__(out_channels, kernel_size, srate, fmin_init, fmax_init, freq, bandwidth, seed=seed)
        self.in_channels = in_channels
        self.pad = TemporalPad(padding='same', dim='2d', kernel_size=kernel_size, padding_mode=padding_mode, hilbert=True)
        self.register_buffer('_hamming_window', torch.hamming_window(kernel_size).reshape((1,1,-1)))
        self.hilbert = HilbertLayer()

    def _create_filters(self, freq_low, freq_high):
        _scale = self._scale.reshape((1,1,1,-1))
        freq_low, freq_high = freq_low.reshape((-1,1,1,1)), freq_high.reshape((-1,1,1,1))   
        filt_low = freq_low * torch.special.sinc(2 * freq_low * self._scale)
        filt_high = freq_high * torch.special.sinc(2 * freq_high * self._scale)
        filt = self._hamming_window * 2 * (filt_high - filt_low) / self.srate
        return filt
        
    def forward(self, x, return_filtered=False):
        x = self.pad(x)
        _, _, freq_low, freq_high = self._create_frequencies()
        filt = self._create_filters(freq_low, freq_high)
        assert self.in_channels == x.shape[-3]
        x = F.conv2d(x, filt, groups=self.in_channels, padding='valid')
            
        if not return_filtered:
            x = self.hilbert(x)
            x = torch.abs(x)
        x = x[...,self.pad.padding_hilbert:-self.pad.padding_hilbert]
        return x


    
    
class WaveletLayer1d(TemporalFilter):
    def __init__(self, in_channels, out_channels, kernel_size, srate, fmin_init, fmax_init, freq=None, bandwidth=None, padding_mode='zeros', seed=None):
        super().__init__(out_channels, kernel_size, srate, fmin_init, fmax_init, freq, bandwidth, seed=seed)
        self.in_channels = in_channels
        self.pad = TemporalPad(padding='same', dim='2d', kernel_size=kernel_size, padding_mode=padding_mode, hilbert=False)     
           
    def _create_filters(self, freq, bandwidth):
        _scale = self._scale.reshape((1,1,-1))
        freq, bandwidth = freq.reshape((-1,1,1)), bandwidth.reshape((-1,1,1))
        sigma2 = (2 * math.log(2)) / (bandwidth * math.pi)**2
        filt = (2 * math.pi * sigma2)**(-1/2) / (self.srate / 2)
        filt = filt * torch.cos(2*math.pi * freq * _scale)
        filt = filt * torch.exp(- _scale**2 / (2 * sigma2))
        return filt
                            
    def forward(self, x):
        x = self.pad(x)
        freq, bandwidth, _, _ = self._create_frequencies()
        filt = self._create_filters(freq, bandwidth)
        assert self.in_channels == x.shape[-2]
        x = F.conv1d(x, filt, groups=self.in_channels, padding='valid')
        return x

    
class WaveletLayer2d(TemporalFilter):
    def __init__(self, in_channels, out_channels, kernel_size, srate, fmin_init, fmax_init, freq=None, bandwidth=None, padding_mode='zeros', seed=None):
        super().__init__(out_channels, kernel_size, srate, fmin_init, fmax_init, freq, bandwidth, seed=seed)
        self.in_channels = in_channels
        self.pad = TemporalPad(padding='same', dim='2d', kernel_size=kernel_size, padding_mode=padding_mode, hilbert=False)     
           
    def _create_filters(self, freq, bandwidth):
        _scale = self._scale.reshape((1,1,1,-1))
        freq, bandwidth = freq.reshape((-1,1,1,1)), bandwidth.reshape((-1,1,1,1))
        sigma2 = (2 * math.log(2)) / (bandwidth * math.pi)**2
        filt = (2 * math.pi * sigma2)**(-1/2) / (self.srate / 2)
        filt = filt * torch.cos(2*math.pi * freq * _scale)
        filt = filt * torch.exp(- _scale**2 / (2 * sigma2))
        return filt
                            
    def forward(self, x):
        x = self.pad(x)
        freq, bandwidth, _, _ = self._create_frequencies()
        filt = self._create_filters(freq, bandwidth)
        assert self.in_channels == x.shape[-3]
        x = F.conv2d(x, filt, groups=self.in_channels, padding='valid')
        return x
        
        
        
        
class ComplexWaveletLayer1d(TemporalFilter):
    def __init__(self, in_channels, out_channels, kernel_size, srate, fmin_init, fmax_init, freq=None, bandwidth=None, padding_mode='zeros', seed=None):
        super().__init__(out_channels, kernel_size, srate, fmin_init, fmax_init, freq, bandwidth, seed=seed)
        self.in_channels = in_channels
        self.pad = TemporalPad(padding='same', dim='1d', kernel_size=kernel_size, padding_mode=padding_mode, hilbert=False)    
           
    def _create_filters(self, freq, bandwidth):
        _scale = self._scale.reshape((1,1,-1))
        freq, bandwidth = freq.reshape((-1,1,1)), bandwidth.reshape((-1,1,1))
        sigma2 = (2 * math.log(2)) / (bandwidth * math.pi)**2
        filt = (2 * math.pi * sigma2)**(-1/2) / (self.srate / 2)
        filt = filt * (torch.exp(1j*2*math.pi * freq * _scale) - torch.exp(-0.5*(2*math.pi * freq)**2))
        filt = filt * torch.exp(- _scale**2 / (2 * sigma2))
        return filt
          
    def forward(self, x, return_filtered=False):
        x = self.pad(x)
        freq, bandwidth, _, _ = self._create_frequencies()
        filt = self._create_filters(freq, bandwidth)
        assert self.in_channels == x.shape[-2]
        
        if return_filtered:
            x = F.conv1d(x, filt.real, groups=self.in_channels, padding='valid')
        else:
            x = x.to(torch.complex64)
            x = F.conv1d(x, filt, groups=self.in_channels, padding='valid')
            x = torch.abs(x)
        return x
    
    
class ComplexWaveletLayer2d(TemporalFilter):
    def __init__(self, in_channels, out_channels, kernel_size, srate, fmin_init, fmax_init, freq=None, bandwidth=None, padding_mode='zeros', seed=None):
        super().__init__(out_channels, kernel_size, srate, fmin_init, fmax_init, freq, bandwidth, seed=seed)
        self.in_channels = in_channels
        self.pad = TemporalPad(padding='same', dim='2d', kernel_size=kernel_size, padding_mode=padding_mode, hilbert=False)    
           
    def _create_filters(self, freq, bandwidth):
        _scale = self._scale.reshape((1,1,1,-1))
        freq, bandwidth = freq.reshape((-1,1,1,1)), bandwidth.reshape((-1,1,1,1))
        sigma2 = (2 * math.log(2)) / (bandwidth * math.pi)**2
        filt = (2 * math.pi * sigma2)**(-1/2) / (self.srate / 2)
        filt = filt * (torch.exp(1j*2*math.pi * freq * _scale) - torch.exp(-0.5*(2*math.pi * freq)**2))
        filt = filt * torch.exp(- _scale**2 / (2 * sigma2))
        return filt
          
    def forward(self, x, return_filtered=False):
        x = self.pad(x)
        freq, bandwidth, _, _ = self._create_frequencies()
        filt = self._create_filters(freq, bandwidth)
        
        assert self.in_channels == x.shape[-3]
        if return_filtered:
            x = F.conv2d(x, filt.real, groups=self.in_channels, padding='valid')
        else:
            x = x.to(torch.complex64)
            x = F.conv2d(x, filt, groups=self.in_channels, padding='valid')
            x = torch.abs(x)
        return x