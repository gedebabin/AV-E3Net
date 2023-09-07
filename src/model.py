import lightning.pytorch as pl
import torch
from torchinfo import summary
from torch import Tensor, nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.save_fig import save_fig
from utils.plot_waveform import plot_waveform
from typing import Tuple
from shufflenet_encoder import _shufflenetv2_05


class ProjectionBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.PReLU(),
            nn.LayerNorm(out_features)
        )

    def forward(self, x):
        return self.block(x)


class GSFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.projection_block = ProjectionBlock(512, 512)
        self.gate = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Sigmoid()
        )
        self.layernorm = nn.LayerNorm(512)

    def forward(self, x, dense_audio):
        vx, ax = x
        # print('GS input', 'audio', ax.shape, 'video', vx.shape)
        # vx = F.interpolate(vx.transpose(1, 2), ax.size(1)).transpose(1, 2)
        vx = torch.cat([ax, vx], dim=2)
        vx = self.gate(vx)
        dense_audio += ax
        ax *= vx
        ax = self.projection_block(ax)
        ax += dense_audio
        ax = self.layernorm(ax)
        # print('GS output', ax.shape)
        return ax, dense_audio


class LSTMBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.gsfusion = GSFusion()

        self.map_to_high_dim_a = nn.Sequential(nn.Linear(512, 1024), nn.PReLU(),
                                               nn.Linear(1024, 512), nn.PReLU(), nn.LayerNorm(512))
        self.map_to_high_dim_v = nn.Sequential(nn.Linear(512, 1024), nn.PReLU(),
                                               nn.Linear(1024, 512), nn.PReLU(), nn.LayerNorm(512))

        self.lstm_a = nn.LSTM(512, 512)
        self.layer_norm_a = nn.LayerNorm(512)
        self.layer_norm_a2 = nn.LayerNorm(512)
        self.lstm_v = nn.LSTM(512, 512)
        self.layer_norm_v = nn.LayerNorm(512)
        self.layer_norm_v2 = nn.LayerNorm(512)

    def forward(self, x, dense_audio, dense_video):
        vx, ax = x
        ax, dense_audio = self.gsfusion((vx, ax), dense_audio)

        ax = self.map_to_high_dim_a(ax)
        dense_audio += ax
        ax, h = self.lstm_a(ax)
        ax = self.layer_norm_a(ax)
        ax += dense_audio
        ax = self.layer_norm_a2(ax)

        vx = self.map_to_high_dim_v(vx)
        dense_video += vx
        vx, h = self.lstm_v(vx)
        vx = self.layer_norm_v(vx)
        vx += dense_video
        vx = self.layer_norm_v2(vx)

        return (vx, ax), dense_audio, dense_video


class AVE3Net(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # audio
        self.window = 320
        self.stride = 160
        self.encoder = nn.Conv1d(1, 2048, kernel_size=self.window, stride=self.stride)
        self.l1 = nn.Sequential(nn.PReLU(), nn.LayerNorm(2048))
        self.audio_projection_block = ProjectionBlock(2048, 512)
        self.mask_prediction = nn.Sequential(nn.Linear(512, 2048), nn.Sigmoid())
        self.decoder = nn.ConvTranspose1d(2048, 1, self.window, self.stride)
        #############

        # video
        self.shufflenet = _shufflenetv2_05()
        self.video_projection_block = ProjectionBlock(1024, 512)

        # GS fusion
        self.gsfusion = GSFusion()

        # LSTM
        self.lstm1 = LSTMBlock()
        self.lstm2 = LSTMBlock()
        self.lstm3 = LSTMBlock()
        self.lstm4 = LSTMBlock()

    def pad_signal(self, input):
        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.window - (self.stride + nsample %
                              self.window) % self.window
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(
            batch_size, 1, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    # def on_after_backward(self) -> None:
    #     print("on_before_opt enter")
    #     for name, p in self.named_parameters():
    #         if p.grad is None:
    #             print(name)
    #     print("on_before_opt exit")

    def forward(self, x: Tuple[Tensor, Tensor]):
        """
        x of shape (vx, ax)

        vx 5D (batch_size, frames, channels, width, height) or 4D without batch_size

        ax 3D (batch_size, channels=1, samples) or 2D without batch_size
        """

        vx, ax = x
        # print(vx.shape, ax.shape)

        if vx.dim() not in [4, 5]:
            raise RuntimeError(f"AV-E3Net video input wrong shape: {vx.shape}")

        if ax.dim() not in [2, 3]:
            raise RuntimeError(f"AV-E3Net audio input wrong shape: {ax.shape}")

        # add minibatch dim to inputs
        if vx.dim() == 4:
            vx = vx.unsqueeze(0)
        if ax.dim() == 2:
            ax = ax.unsqueeze(0)

        # audio

        ax, rest = self.pad_signal(ax)
        audio_encoded = self.encoder(ax)
        # print('audion_encoded', audio_encoded.shape)

        ax = audio_encoded.transpose(1, 2)
        ax = self.l1(ax)
        ax = self.audio_projection_block(ax)

        ##############

        # video

        n_frames = vx.size(1)
        vx = vx.view(-1, 3, 96, 96)  # [4, 71, 3, 96, 96] to [284, 3, 96, 96]
        vx = self.shufflenet(vx)
        vx = vx.view(-1, n_frames, 1024)  # transform back
        vx = self.video_projection_block(vx)
        # print('after projection blocks', 'audio', ax.shape, 'video', vx.shape)
        ##############

        # LSMT blocks

        # upsample video
        vx = F.interpolate(vx.transpose(1, 2), ax.size(1)).transpose(1, 2)
        # initialize dense sums
        dense_audio = torch.zeros_like(ax)
        dense_video = torch.zeros_like(vx)

        (vx, ax), dense_audio, dense_video = self.lstm1((vx, ax), dense_audio, dense_video)
        (vx, ax), dense_audio, dense_video = self.lstm2((vx, ax), dense_audio, dense_video)
        (vx, ax), dense_audio, dense_video = self.lstm3((vx, ax), dense_audio, dense_video)
        (vx, ax), dense_audio, dense_video = self.lstm4((vx, ax), dense_audio, dense_video)

        ##############

        # final stage
        ax, dense_audio = self.gsfusion((vx, ax), dense_audio)
        ax = self.mask_prediction(ax)
        ax = ax.transpose(1, 2)

        # print('after sigmoid', ax.shape)

        ax = ax * audio_encoded  # why ax *= audio_encoded does not work??

        ax = self.decoder(ax)
        ax = ax[:, :, self.stride: -(rest + self.stride)].contiguous()  # B*C, 1, L

        ##############

        return ax

    def training_step(self, batch):
        video, noisy, clean = batch
        # print('lens', len(video), len(noisy), len(clean))
        # print('ts video', video[0].shape, video[1].shape)
        # print('ts noisy', noisy[0].shape, noisy[1].shape)
        # print('ts clean', clean[0].shape, clean[1].shape)

        video = nn.utils.rnn.pad_sequence(video, batch_first=True)  # pad batch to the max size
        noisy = nn.utils.rnn.pad_sequence(noisy, batch_first=True).transpose(1, 2)
        clean = nn.utils.rnn.pad_sequence(clean, batch_first=True).transpose(1, 2)

        # print('padded video', video.shape)
        # print('padded noisy', noisy.shape)
        # print('padded clean', clean.shape)

        x_hat = self.forward((video, noisy))
        loss = nn.functional.mse_loss(x_hat, clean)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    data = [(torch.rand((16, 25, 3, 96, 96)), torch.rand((16, 1, 16000)))]
    summary(AVE3Net(), input_data=data, col_names=["input_size",
                                                   "output_size",
                                                   "num_params"], depth=2)
    # summary(AVE3Net(), input_size=(1, 60000))
