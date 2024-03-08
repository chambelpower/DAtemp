import random
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from speechbrain.dataio.legacy import ExtendedCSVDataset
from speechbrain.dataio.dataloader import make_dataloader
from speechbrain.processing.signal_processing import (
    compute_amplitude,
    dB_to_amplitude,
    convolve1d,
    notch_filter,
    reverberate,
)
import matplotlib.pyplot as plt

class AddVariousNoises(torch.nn.Module):
    """This class adds various types of noises to the input signal.

    Arguments
    ---------
    noise_types : list
        List of noise types to add, e.g., ['gaussian', 'poisson', 'salt_and_pepper'].
    noise_params : dict
        Dictionary of noise parameters for each type, e.g., {'gaussian': {'mean': 0, 'std': 0.1}, 'poisson': {}, 'salt_and_pepper': {'prob': 0.05}}.
    """

    def __init__(self, noise_types, noise_params):
        super().__init__()

        self.noise_types = noise_types
        self.noise_params = noise_params

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """
        noisy_waveform = waveforms.clone()

        for noise_type in self.noise_types:
            if noise_type == 'gaussian':
                mean = self.noise_params.get(noise_type, {}).get('mean', 0)
                std = self.noise_params.get(noise_type, {}).get('std', 0.1)
                noise = torch.randn_like(waveforms) * std + mean

            elif noise_type == 'poisson':
                # Poisson noise is often applied directly to the signal without additional parameters
                noise = torch.poisson(waveforms)

            elif noise_type == 'salt_and_pepper':
                prob = self.noise_params.get(noise_type, {}).get('prob', 0.05)
                noise = F.dropout(waveforms, p=prob, training=self.training, inplace=False)

            # Add more noise types as needed...

            else:
                raise ValueError(f"Unsupported noise type: {noise_type}")

            noisy_waveform += noise

        return noisy_waveform

# Example usage:
noise_adder = AddVariousNoises(
    noise_types=['gaussian', 'poisson', 'salt_and_pepper'],
    noise_params={'gaussian': {'mean': 0, 'std': 0.1}, 'poisson': {}, 'salt_and_pepper': {'prob': 0.05}}
)

clean_signal = torch.randn((1, 100))  # Example clean signal
print(clean_signal)

noisy_signal = noise_adder(clean_signal)
print(noisy_signal)

# Plot original and augmented signals
plt.figure(figsize=(12, 6))

 # Original signal
plt.subplot(2, num_examples, 1)
plt.plot(clean_signal.squeeze().numpy())
plt.title(f"Original Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")

# Augmented signal
noisy_signal = noise_adder(clean_signal)
plt.subplot(2, num_examples, 2)
plt.plot(noisy_signal.squeeze().numpy())
plt.title(f"Augmented Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()