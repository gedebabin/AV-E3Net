import matplotlib.pyplot as plt
import torch


def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show()


def plot_2_waveform(
    waveform1, waveform2, sample_rate, title="Waveform", xlim=None, ylim=None
):
    if waveform1.shape != waveform2.shape:
        raise ValueError()

    num_channels, num_frames = waveform1.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 2)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c][0].plot(time_axis, waveform1[c], linewidth=1)
        axes[c][0].grid(True)
        if num_channels > 1:
            axes[c][0].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c][0].set_xlim(xlim)
        if ylim:
            axes[c][0].set_ylim(ylim)

    for c in range(num_channels):
        axes[c][1].plot(time_axis, waveform2[c], linewidth=1)
        axes[c][1].grid(True)
        if num_channels > 1:
            axes[c][1].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c][1].set_xlim(xlim)
        if ylim:
            axes[c][1].set_ylim(ylim)
    figure.suptitle(title)
    figure.savefig('figure.png')


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
