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

class ElectrodeDisplacement:
    def __init__(self, displacement_factor=0.1):
        """
        Electrode Displacement Data Augmentation.

        Parameters:
        - displacement_factor (float): Proportion of electrode displacement relative to the total timepoints.
        """
        self.displacement_factor = displacement_factor

    def forward(self, eeg_data):
        """
        Apply electrode displacement augmentation.

        Parameters:
        - eeg_data (torch.Tensor): Input EEG data with shape (batch_size, num_channels, num_timepoints).

        Returns:
        - augmented_data (torch.Tensor): Augmented EEG data.
        """
        batch_size, num_channels, num_timepoints = eeg_data.size()

        # Calculate the number of timepoints to displace
        displacement_steps = int(self.displacement_factor * num_timepoints)

        # Generate random displacement indices for each sample in the batch
        displacement_indices = torch.randint(
            low=-displacement_steps, high=displacement_steps + 1, size=(batch_size,)
        )

        # Apply electrode displacement
        augmented_data = torch.zeros_like(eeg_data)
        for i in range(batch_size):
            augmented_data[i, :, :] = F.roll(eeg_data[i, :, :], shifts=displacement_indices[i], dims=-1)

        return augmented_data

# Example usage:
# Assuming eeg_data is a PyTorch tensor with shape (batch_size, num_channels, num_timepoints)
eeg_augmentor = ElectrodeDisplacement(displacement_factor=0.1)
augmented_eeg_data = eeg_augmentor.forward(eeg_data)

