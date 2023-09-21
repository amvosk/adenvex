import unittest
from envelope_detector import HilbertLayer

import numpy as np
import scipy.signal as sg
import torch


class TestLayers(unittest.TestCase):
    def test_hilbert_layer(self):
        hilbert_layer = HilbertLayer()
        for seed in range(200, 300):
#             self.assertEqual(module1.function1(…), expected_value)
            torch.manual_seed(seed)
            x = torch.randn(100, seed)
            x_hilbert_1 = sg.hilbert(x.numpy(), axis=-1)
            with torch.no_grad():
                x_hilbert_2 = hilbert_layer(x).numpy()
            value = np.linalg.norm(x_hilbert_1 - x_hilbert_2) / np.sqrt(seed * 100)
            self.assertAlmostEqual(value, 0, places=5)