import lightning.pytorch as pl
from torch.utils.data import DataLoader
from e3net.dataset_e3net import DatasetAudioOnly
from dataset import Dataset

AUDIO_ONLY = 0


def collate_fn(batch):
    if AUDIO_ONLY:
        # audio-only
        x = [item[0].transpose(0, 1) for item in batch]
        y = [item[1].transpose(0, 1) for item in batch]
        return x, y

    video = [item[0][0] for item in batch]
    noisy = [item[0][1].transpose(0, 1) for item in batch]
    clean = [item[1].transpose(0, 1) for item in batch]
    return video, noisy, clean


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 1):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: str):

        ######################
        if AUDIO_ONLY:
            if stage == "fit":
                self.train = DatasetAudioOnly('train_avgen.tsv')
                self.valid = DatasetAudioOnly('valid_avgen.tsv')

            if stage == 'test':
                self.test = DatasetAudioOnly('test_avgen.tsv')

            return
        ######################

        if stage == "fit":
            self.train = Dataset('train_avgen.tsv')
            self.valid = Dataset('valid_avgen.tsv')

        if stage == 'test':
            self.test = Dataset('test_avgen.tsv')

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=12, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size, num_workers=12, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=12, collate_fn=collate_fn)

    # def teardown(self, stage: str):
    #     # Used to clean-up when the run is finished
    #     pass


if __name__ == "__main__":
    datamodule = DataModule(batch_size=4)
    datamodule.setup("fit")

    train_dataloader = datamodule.train_dataloader()

    i = iter(train_dataloader)

    batch = next(i)
    x, y = batch

    if AUDIO_ONLY:
        print(x[0].shape)
    else:
        v, a = x[0]
        print(v.shape, a.shape)

    # print(x.shape, y.shape)

    # print(len(train_dataloader.dataset))

    # noisy, clean = train_dataloader.dataset[0]

    # plot_2_waveform(noisy, clean, 16000)
