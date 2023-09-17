import torch
import torchaudio
import torchvision
import utils.logger as logger
from typing import Tuple


class Dataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.logger = logger.get_logger(self.__class__.__name__, logger.logging.NOTSET)
        self.ROOT = '/data/LRS3_30h/'
        self.filename = filename  # test_avgen.tsv

        with open(self.ROOT + self.filename, 'r') as f:
            lines = f.readlines()[1:]  # skip first header line

            # [['test/{id}/00001', 'test/video/{id}/00001.mp4', 'test/clean/{id}/00001.wav', 'test/noisy/{id}/00001.wav', '9.24', '149', '96256'], ...]
            data = list(map(lambda line: line.replace('\n', '').split('\t'), lines))
            # {'id': 'train/{id}/50002', 'video': 'train/video/{id}/50002.mp4', 'clean': 'train/clean/{id}/50002.wav', 'noisy': 'train/noisy/{id}/50002.wav'}
            self.data = [{"id": x[0], "video": x[1], "clean": x[2], "noisy": x[3]} for x in data]
            self.logger.debug(f'self.data[0] {self.data[0]}')

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        video_path = self.ROOT + self.data[i]['video']
        noisy_path = self.ROOT + self.data[i]['noisy']
        clean_path = self.ROOT + self.data[i]['clean']

        vframes, _aframes, _info = torchvision.io.read_video(
            video_path, pts_unit='sec', output_format='TCHW')  # pts_init to avoid warning
        clean_waveform, _sr = torchaudio.load(clean_path)
        noisy_waveform, _sr = torchaudio.load(noisy_path)

        self.logger.debug(f'vframes.shape: {vframes.shape}')  # vframes.shape: [80, 3, 96, 96]
        self.logger.debug(f'clean_waveform.shape: {clean_waveform.shape}')  # clean_waveform.shape: [1, 51200]
        self.logger.debug(f'noisy_waveform.shape: {clean_waveform.shape}')  # noisy_waveform.shape: [1, 51200]

        # normalizetion for shufflenet https://pytorch.org/hub/pytorch_vision_shufflenet_v2/
        # preprocess = torchvision.transforms.Compose([
        #     torchvision.transforms.Resize(224),
        #     torchvision.transforms.CenterCrop(224),
        #     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])
        # vframes = preprocess(vframes)
        vframes = vframes / 255

        return vframes, noisy_waveform, clean_waveform

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = Dataset('train_avgen.tsv')
    vframes, noisy, clean = dataset[20]
