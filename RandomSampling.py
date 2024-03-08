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

class RandomSampling:
    def __init__(self, sampling_factor=0.1):
        """
        Random Sampling Data Augmentation.

        Parameters:
        - sampling_factor (float): Proportion of timepoints to randomly sample.
        """
        self.sampling_factor = sampling_factor

    def forward(self, eeg_data):
        """
        Apply random sampling augmentation.

        Parameters:
        - eeg_data (torch.Tensor): Input EEG data with shape (batch_size, num_channels, num_timepoints).

        Returns:
        - augmented_data (torch.Tensor): Augmented EEG data.
        """
        batch_size, num_channels, num_timepoints = eeg_data.size()

        # Calculate the number of timepoints to randomly sample
        sampled_timepoints = int(self.sampling_factor * num_timepoints)

        # Generate random indices for each sample in the batch
        sampled_indices = torch.randint(low=0, high=num_timepoints, size=(batch_size, sampled_timepoints))

        # Create a mask for sampled timepoints
        mask = torch.zeros_like(eeg_data)
        for i in range(batch_size):
            mask[i, :, sampled_indices[i, :]] = 1.0

        # Apply random sampling
        augmented_data = eeg_data * mask

        return augmented_data

# Example usage:
# Assuming eeg_data is a PyTorch tensor with shape (batch_size, num_channels, num_timepoints)
eeg_augmentor = RandomSampling(sampling_factor=0.1)
augmented_eeg_data = eeg_augmentor.forward(eeg_data)