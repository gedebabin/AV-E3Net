import lightning.pytorch as pl
from torch.utils.data import DataLoader
from dataset_e3net import DatasetAudioOnly


def collate_fn(batch):
    x = [item[0].transpose(0, 1) for item in batch]
    y = [item[1].transpose(0, 1) for item in batch]
    print('collate_fn', x[0].shape, y[0].shape, len(x), len(y))
    return x, y


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 1):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: str):
        if stage == "fit":
            self.train = DatasetAudioOnly('train_avgen.tsv')
            self.valid = DatasetAudioOnly('valid_avgen.tsv')

        if stage == 'test':
            self.test = DatasetAudioOnly('test_avgen.tsv')

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=12, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size, num_workers=12, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=12, collate_fn=collate_fn)


if __name__ == "__main__":
    datamodule = DataModule(batch_size=4)
    datamodule.setup("fit")

    train_dataloader = datamodule.train_dataloader()

    i = iter(train_dataloader)

    batch = next(i)
    x, y = batch

    print(x[0].shape)

    # print(x.shape, y.shape)

    # print(len(train_dataloader.dataset))

    # noisy, clean = train_dataloader.dataset[0]

    # plot_2_waveform(noisy, clean, 16000)
