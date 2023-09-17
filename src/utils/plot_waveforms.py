from typing import List, Tuple
import torch
import matplotlib.pyplot as plt


def plot_waveforms(sample_rate, items: List[Tuple[torch.Tensor, str]]):
    (first_waveform, _title) = items[0]

    if not all([waveform.shape == first_waveform.shape for (waveform, _title) in items]):
        raise ValueError("Different shapes")

    num_channels, num_frames = first_waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    for i, (waveform, title) in enumerate(items):
        plt.subplot(len(items), 1, i+1)
        plt.plot(time_axis, waveform[0])
        plt.title(title)

    plt.subplots_adjust(hspace=0.75)
    plt.savefig('figure.png')


# def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
#     waveform = waveform.numpy()

#     num_channels, num_frames = waveform.shape
#     time_axis = torch.arange(0, num_frames) / sample_rate

#     figure, axes = plt.subplots(num_channels, 1)
#     if num_channels == 1:
#         axes = [axes]
#     for c in range(num_channels):
#         axes[c].specgram(waveform[c], Fs=sample_rate)
#         if num_channels > 1:
#             axes[c].set_ylabel(f"Channel {c+1}")
#         if xlim:
#             axes[c].set_xlim(xlim)
#     figure.suptitle(title)
#     plt.show()
