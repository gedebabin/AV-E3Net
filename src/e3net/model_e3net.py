import lightning.pytorch as pl
import torch
from torchinfo import summary
from torch import nn, Tensor
from torch.autograd import Variable
from typing import List, Tuple
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio, perceptual_evaluation_speech_quality, signal_distortion_ratio
from utils import logger




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


class LSTMBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.map_to_high_dim_a = nn.Sequential(nn.Linear(512, 1024), nn.PReLU(),
                                               nn.Linear(1024, 512), nn.PReLU(), nn.LayerNorm(512))

        self.lstm = nn.LSTM(512, 512)
        self.layer_norm1 = nn.LayerNorm(512)
        self.layer_norm2 = nn.LayerNorm(512)

    def forward(self, x, ha):
        ax = x

        ax_before_lstm = self.map_to_high_dim_a(ax)
        if (ha != None):
            ax, ha = self.lstm(ax_before_lstm, ha)
        else:
            ax, ha = self.lstm(ax_before_lstm)
        ax = self.layer_norm1(ax)

        ax += ax_before_lstm
        ax = self.layer_norm2(ax)

        return ax, ha


class E3NetModule(nn.Module):
    def __init__(self, n_lstm=4):
        super().__init__()
        self.debug_logger = logger.get_logger(self.__class__.__name__, logger.logging.NOTSET)

        self.ha = None

        # audio
        self.window = 320
        self.stride = 160
        self.encoder = nn.Conv1d(1, 2048, kernel_size=self.window, stride=self.stride)
        self.l1 = nn.Sequential(nn.PReLU(), nn.LayerNorm(2048))
        self.audio_projection_block = ProjectionBlock(2048, 512)

        self.mask_prediction = nn.Sequential(nn.Linear(512, 2048), nn.Sigmoid())
        self.decoder = nn.ConvTranspose1d(2048, 1, self.window, self.stride)

        # LSTM
        self.lstm_blocks = nn.ModuleList()
        for _ in range(n_lstm):
            self.lstm_blocks.append(LSTMBlock())

    def pad_signal(self, input):
        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.window - (self.stride + nsample % self.window) % self.window
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, 1, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    # def on_after_backward(self) -> None:
    #     print("on_before_opt enter")
    #     for name, p in self.named_parameters():
    #         if p.grad is None:
    #             print(name)
    #     print("on_before_opt exit")

    def forward(self, x: Tensor) -> Tensor:
        """
        x of shape ax 3D (batch_size, channels=1, samples) or 2D without batch_size
        """

        ax = x
        self.debug_logger.debug(f'forward input ax.shape {ax.shape}')

        if ax.dim() not in [2, 3]:
            raise RuntimeError(f"AV-E3Net audio input wrong shape: {ax.shape}")

        # add minibatch dim to inputs
        if ax.dim() == 2:
            ax = ax.unsqueeze(0)

        ax, rest = self.pad_signal(ax)
        audio_encoded = self.encoder(ax)
        self.debug_logger.debug(f'audio_encoded.shape {audio_encoded.shape}')

        ax = audio_encoded.transpose(1, 2)
        ax = self.l1(ax)
        self.debug_logger.debug(f'ax.shape {ax.shape}')
        ax = self.audio_projection_block(ax)
        self.debug_logger.debug(f'ax.shape {ax.shape}')

        # LSMT blocks
        ha = self.ha
        for lstm in self.lstm_blocks:
            ax, ha = lstm(ax, ha)
        self.ha = ha
        ##############

        # final stage
        ax = self.mask_prediction(ax)
        ax = ax.transpose(1, 2)

        ax = ax * audio_encoded

        ax = self.decoder(ax)
        ax = ax[:, :, self.stride: -(rest + self.stride)].contiguous()  # B*C, 1, L

        ##############

        return ax


class E3Net(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.debug_logger = logger.get_logger(self.__class__.__name__, logger.logging.NOTSET)
        self.e3net = E3NetModule(4)

    def forward(self, x: Tensor) -> Tensor:
        return self.e3net(x)

    def process_batch(self, batch: Tuple[List[Tensor], List[Tensor]], batch_idx):
        noisy, clean = batch
        self.debug_logger.debug(f'process_batch input noisy[0]{noisy[0].shape}')

        # convert audios from [1, T] to [T, 1] for pad_sequence
        noisy = [x.transpose(0, 1) for x in noisy]
        clean = [x.transpose(0, 1) for x in clean]

        # pad batch to max size
        # audios transposed back to [1, T] after padding
        noisy = nn.utils.rnn.pad_sequence(noisy, batch_first=True).transpose(1, 2)
        clean = nn.utils.rnn.pad_sequence(clean, batch_first=True).transpose(1, 2)
        # padded noisy [16, 1, 98304], clean [16, 1, 98304]
        self.debug_logger.debug(f'padded noisy {noisy.shape}, clean {clean.shape}')

        x_hat = self.forward(noisy)
        return x_hat, clean

    def training_step(self, batch: Tuple[List[Tensor], List[Tensor]], batch_idx):
        self.ha = None

        x_hat, clean = self.process_batch(batch, batch_idx)
        loss = nn.functional.mse_loss(x_hat, clean)
        self.log("train_loss", loss, prog_bar=True, batch_size=16)
        return loss

    def validation_step(self, batch: Tuple[List[Tensor], List[Tensor]], batch_idx):
        self.ha = None

        x_hat, clean = self.process_batch(batch, batch_idx)
        loss = nn.functional.mse_loss(x_hat, clean)
        self.log("validation_loss", loss, prog_bar=True, sync_dist=True, batch_size=16)
        return loss

    def test_step(self, batch: Tuple[List[Tensor], List[Tensor]], batch_idx):
        self.ha = None

        x_hat, clean = self.process_batch(batch, batch_idx)

        pesq = perceptual_evaluation_speech_quality(x_hat, clean, 16000, 'wb')
        sdr = signal_distortion_ratio(x_hat, clean)
        sisdr = scale_invariant_signal_distortion_ratio(x_hat, clean)
        loss = nn.functional.mse_loss(x_hat, clean)

        log = {
            "loss": loss,
            "pesq": pesq.mean(),
            "sdr": sdr.mean(),
            "sisdr": sisdr.mean()
        }

        self.log_dict(log, prog_bar=True, batch_size=16)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    data = [torch.rand((16, 1, 16000))]
    summary(E3Net(), input_data=data, col_names=["input_size",
                                                 "output_size",
                                                 "num_params"], depth=3)
