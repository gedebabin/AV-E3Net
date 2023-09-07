import torchaudio
from datamodule import DataModule
from utils.plot_waveform import plot_2_waveform
from model import Model
import torch

datamodule = DataModule()
datamodule.setup("test")

train_dataloader = datamodule.test_dataloader()
# model = Model.load_from_checkpoint(
#     checkpoint_path="lightning_logs/version_56/checkpoints/checkpoint.ckpt",
#     map_location=torch.device('cpu')
# )

model = Model()

noisy, clean = train_dataloader.dataset[10]
x_hat = model(noisy)

x_hat = x_hat.detach()


torchaudio.save("x_hat.wav", x_hat[0], 16000)
torchaudio.save("clean.wav", clean, 16000)

plot_2_waveform(x_hat[0], clean, 16000)
