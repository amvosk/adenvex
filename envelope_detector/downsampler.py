import torch
import torch.nn as nn
import torchaudio.transforms

class AvePoolLayer(nn.Module):
    def __init__(self, downsample_coef, window=None):
        super().__init__()
        window = downsample_coef if window is None else window
        self.downsampler = nn.AvgPool1d(kernel_size=window, stride=downsample_coef)

    def forward(self, x):
        x = self.downsampler(x)
        return x
    
class ResampleLayer(nn.Module):
    def __init__(self, downsample_coef):
        super().__init__()
        self.downsampler = torchaudio.transforms.Resample(downsample_coef, 1)

    def forward(self, x):
        x = self.downsampler(x)
        return x