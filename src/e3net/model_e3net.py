import lightning.pytorch as pl
import torch
from torchinfo import summary
from torch import nn
from torch.autograd import Variable


class LSTMBlock(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.map_to_high_dim_a = nn.Sequential(nn.Linear(512, 1024), nn.PReLU(),
                                               nn.Linear(1024, 512), nn.PReLU(), nn.LayerNorm(512))
        self.map_to_high_dim_v = nn.Sequential(nn.Linear(512, 1024), nn.PReLU(),
                                               nn.Linear(1024, 512), nn.PReLU(), nn.LayerNorm(512))

        self.lstm_a = nn.LSTM(512, 512)
        self.layer_norm_a = nn.LayerNorm(512)
        self.lstm_v = nn.LSTM(512, 512)
        self.layer_norm_v = nn.LayerNorm(512)

    def forward(self, x):
        vx, ax = x
        ax = self.gsfusion((vx, ax))

        ax = self.map_to_high_dim_a(ax)
        # add to dense sum a
        ax, h = self.lstm_a(ax)
        ax = self.layer_norm_a(ax)
        # add dense sum a
        # add layernorm

        vx = self.map_to_high_dim_a(vx)
        # add to dense sum v
        vx, h = self.lstm_a(vx)
        vx = self.layer_norm_v(vx)
        # add dense sum v
        # add layernorm

        return (vx, ax)


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


class ModelAudioOnly(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.window = 320
        self.stride = 160

        self.encoder = nn.Conv1d(1, 2048, kernel_size=self.window, stride=self.stride)
        self.l1 = nn.Sequential(nn.PReLU(), nn.LayerNorm(2048))
        # self.speaker_embedding = None
        self.l2 = ProjectionBlock(2048, 1024)
        self.mask = nn.Sequential(nn.Linear(1024, 2048), nn.Sigmoid())
        self.decoder = nn.ConvTranspose1d(2048, 1, self.window, self.stride)

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

    def forward(self, x):
        print('forward', x.shape)

        x, rest = self.pad_signal(x)
        x = self.encoder(x)

        x = x.transpose(1, 2)
        x = self.l1(x)
        x = self.l2(x)
        x = self.mask(x)
        x = x.transpose(1, 2)

        x = self.decoder(x)
        x = x[:, :, self.stride: -(rest + self.stride)].contiguous()  # B*C, 1, L
        return x

    def training_step(self, batch):
        x = nn.utils.rnn.pad_sequence(batch[0], batch_first=True).transpose(1, 2)
        y = nn.utils.rnn.pad_sequence(batch[1], batch_first=True).transpose(1, 2)
        x_hat = self.forward(x)
        loss = nn.functional.mse_loss(x_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    summary(ModelAudioOnly(), input_size=((64, 60000)))
