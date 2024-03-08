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

class ChannelPermutation:
    def __init__(self, permutation_prob=0.1):
        """
        Channel Permutation Data Augmentation.

        Parameters:
        - permutation_prob (float): Probability of permuting the order of channels.
        """
        self.permutation_prob = permutation_prob

    def forward(self, eeg_data):
        """
        Apply channel permutation augmentation.

        Parameters:
        - eeg_data (torch.Tensor): Input EEG data with shape (batch_size, num_channels, num_timepoints).

        Returns:
        - augmented_data (torch.Tensor): Augmented EEG data.
        """
        batch_size, num_channels, num_timepoints = eeg_data.size()

        # Check if permutation should be applied
        if torch.rand(1).item() < self.permutation_prob:
            # Generate a random permutation of channel indices
            permutation_indices = torch.randperm(num_channels)

            # Apply permutation to each sample in the batch
            augmented_data = eeg_data[:, permutation_indices, :]

            return augmented_data
        else:
            # If no permutation, return the original data
            return eeg_data

# Example usage:
# Assuming eeg_data is a PyTorch tensor with shape (batch_size, num_channels, num_timepoints)
eeg_augmentor = ChannelPermutation(permutation_prob=0.1)
augmented_eeg_data = eeg_augmentor.forward(eeg_data)