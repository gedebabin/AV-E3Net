import lightning.pytorch as pl
from torch.utils.data import DataLoader
from dataset import Dataset
from utils.plot_waveforms import plot_waveforms
import utils.logger as logger
from typing import List, Tuple
from torch import Tensor


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 1):
        super().__init__()
        self.logger = logger.get_logger(self.__class__.__name__, logger.logging.NOTSET)
        self.batch_size = batch_size

    def setup(self, stage: str):
        if stage == "fit":
            self.train = Dataset('train_avgen.tsv')
            self.valid = Dataset('valid_avgen.tsv')

        if stage == 'test':
            self.test = Dataset('test_avgen.tsv')

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=12, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size, num_workers=12, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=12, collate_fn=self.collate_fn)

    def collate_fn(self, batch: List[Tuple[Tensor, Tensor, Tensor]]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        '''
            Because train/test items are of different length, they need to be regrouped before passing to the model to avoid different sizes in a batch.
            Dataset returns items of type: (vframes, noisy_waveform, clean_waveform)
            1st item e.x.: (vframes [80, 3, 96, 96], noisy_waveform [1, 51200], clean_waveform [1, 51200])
            2nd item e.x.: (vframes [37, 3, 96, 96], noisy_waveform [1, 24576], clean_waveform [1, 24576])

            batch
                list (of size batch_size) of tuples returned by dataset (vframes, noisy_waveform, clean_waveform)
            return
                Regrouped tuple of lists
        '''
        # [batch_size, (vframes, noisy, clean)]
        self.logger.debug(
            f'collate input: batch[0][0].shape {batch[0][0].shape}, batch[0][1].shape {batch[0][1].shape}, batch[0][2].shape {batch[0][2].shape}')
        video = [item[0] for item in batch]
        noisy = [item[1] for item in batch]
        clean = [item[2] for item in batch]
        self.logger.debug(f'collate output: {video[0].shape} {noisy[0].shape} {clean[0].shape}')
        return video, noisy, clean


if __name__ == "__main__":
    datamodule = DataModule(batch_size=2)
    datamodule.setup("fit")

    train_dataloader = datamodule.train_dataloader()
    i = iter(train_dataloader)
    batch: Tuple[List[Tensor], List[Tensor], List[Tensor]] = next(i)
    vframes_list, noisy_list, clean_list = batch

    vframes = vframes_list[0]
    noisy_waveform = noisy_list[0]
    clean_waveform = clean_list[0]

    # plot_2_waveform(noisy_waveform, clean_waveform, 16000, 'noisy', 'clean')
