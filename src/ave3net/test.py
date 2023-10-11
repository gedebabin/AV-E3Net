import torchaudio
from datamodule import DataModule
from utils.plot_waveforms import plot_waveforms
from model import AVE3Net
import torch
from torchvision import transforms
import time
import torch.nn.functional as F


datamodule = DataModule()
datamodule.setup("test")

train_dataloader = datamodule.test_dataloader()
model = AVE3Net.load_from_checkpoint(
    checkpoint_path="lightning_logs/version_86/checkpoints/checkpoint.ckpt",
    map_location=torch.device('cpu')
)


# model = AVE3Net()

# x = (video, noisy_audio)
vframes, noisy, clean = train_dataloader.dataset[205]

# preprocess = transforms.Compose([
#     # transforms.Resize(256),
#     # transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# chunk for chunk 160 samples 1 frame
# v = preprocess(v)


def process_with_rtf(model: AVE3Net, vframes: torch.Tensor, noisy: torch.Tensor):
    print('process_with_rtf input', vframes.shape, noisy.shape)

    # 640 samples = 40ms = 1 video frame. 16kHz/25fps
    # 160 samples = 10ms = 0.25 video frame. Every frame processed 4 times.

    

    a = noisy.split(160, 1)[:-1]  # discard last incomplete chunk
    v = F.interpolate(vframes.transpose(0, 3), (vframes.size(-1), len(a))).transpose(0, 3)
    v = v.split(1)
    v = [*v, v[-1]]

    print(len(a), len(v))

    start = time.time()
    x_hat = torch.tensor([])
    for ach, vch in zip(a, v):
        x_hat_ch = model((vch.unsqueeze(0), ach))
        x_hat = torch.cat((x_hat, x_hat_ch[0][0]))
    x_hat = x_hat.unsqueeze(0)

    # x_hat = model((vframes, noisy)).detach()[0]

    print(x_hat.shape)


    end = time.time()
    # x_hat = x_hat.detach()

    elapsed = end - start
    audio_duration = noisy.size(1) / 16000
    rtf = elapsed / audio_duration
    return x_hat, rtf


x_hat, rtf = process_with_rtf(model, vframes, noisy)
print('rtf', rtf)

torchaudio.save("x_hat.wav", x_hat, 16000)
# torchaudio.save("clean.wav", clean, 16000)
# torchaudio.save("noisy.wav", a, 16000)

# plot_waveforms(16000, [(noisy, 'x'), (x_hat, 'x_hat'), (clean, 'clean')])
print('done')
