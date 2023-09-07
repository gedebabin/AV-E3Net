import torchaudio
from datamodule import DataModule
from utils.plot_waveform import plot_2_waveform
from model import AVE3Net
import torch
from torchvision import transforms

datamodule = DataModule()
datamodule.setup("test")

train_dataloader = datamodule.test_dataloader()
model = AVE3Net.load_from_checkpoint(
    checkpoint_path="lightning_logs/version_84/checkpoints/checkpoint.ckpt",
    map_location=torch.device('cpu')
)

# model = AVE3Net()

# x = (video, noisy_audio)
x, clean = train_dataloader.dataset[205]

# preprocess = transforms.Compose([
#     # transforms.Resize(256),
#     # transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

v, a = x

# v = preprocess(v)


x_hat = model(x)

x_hat = x_hat.detach()


torchaudio.save("x_hat.wav", x_hat[0], 16000)
torchaudio.save("clean.wav", clean, 16000)
torchaudio.save("noisy.wav", a, 16000)

plot_2_waveform(x_hat[0], clean, 16000)
