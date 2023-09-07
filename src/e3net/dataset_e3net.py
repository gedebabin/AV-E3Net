import torch
import torchaudio


class DatasetAudioOnly(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.ROOT = '/data/LRS3_30h/'
        self.filename = filename  # test_avgen.tsv

        with open(self.ROOT + self.filename, 'r') as f:
            lines = f.readlines()[1:]  # skip first header line

            # [['test/{id}/00001', 'test/video/{id}/00001.mp4', 'test/clean/{id}/00001.wav', 'test/noisy/{id}/00001.wav', '9.24', '149', '96256'], ...]
            data = list(map(lambda line: line.replace('\n', '').split('\t'), lines))

            self.data = [{"id": x[0], "clean": x[2], "noisy": x[3]} for x in data]

    def __getitem__(self, i):
        noisy_path = self.ROOT + self.data[i]['noisy']
        clean_path = self.ROOT + self.data[i]['clean']

        clean_waveform, _sr = torchaudio.load(clean_path)
        noisy_waveform, _sr = torchaudio.load(noisy_path)

        return noisy_waveform, clean_waveform

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = DatasetAudioOnly('train_avgen.tsv')
    noisy, clean = dataset[0]
    print(noisy.shape, clean.shape)
    print(clean.min(), clean.max())
