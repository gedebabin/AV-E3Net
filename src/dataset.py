import torch
import torchaudio
import torchvision

import matplotlib.pyplot as plt


class Dataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.ROOT = '/data/LRS3_30h/'
        self.filename = filename  # test_avgen.tsv

        with open(self.ROOT + self.filename, 'r') as f:
            lines = f.readlines()[1:]  # skip first header line

            # [['test/{id}/00001', 'test/video/{id}/00001.mp4', 'test/clean/{id}/00001.wav', 'test/noisy/{id}/00001.wav', '9.24', '149', '96256'], ...]
            data = list(map(lambda line: line.replace('\n', '').split('\t'), lines))

            self.data = [{"id": x[0], "video": x[1], "clean": x[2], "noisy": x[3]} for x in data]

    def __getitem__(self, i):
        video_path = self.ROOT + self.data[i]['video']
        noisy_path = self.ROOT + self.data[i]['noisy']
        clean_path = self.ROOT + self.data[i]['clean']

        vframes, _aframes, _info = torchvision.io.read_video(
            video_path, pts_unit='sec', output_format='TCHW')  # pts_init to avoid warning
        clean_waveform, _sr = torchaudio.load(clean_path)
        noisy_waveform, _sr = torchaudio.load(noisy_path)

        # normalizetion for shufflenet https://pytorch.org/hub/pytorch_vision_shufflenet_v2/
        preprocess = torchvision.transforms.Compose([
            # torchvision.transforms.Resize(224),
            # torchvision.transforms.CenterCrop(224),
            # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        vframes = vframes / 255
        vframes = preprocess(vframes)

        # dog = torchvision.io.read_image('dog.jpg')
        # dog = dog / 255
        # dog = preprocess(dog)
        # plt.imsave('vframe.jpg', vframes[10].transpose(0, 1).transpose(1, 2).numpy())
        # print('dog', dog.shape)

        return (vframes, noisy_waveform), clean_waveform

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = Dataset('train_avgen.tsv')
    (vframes, noisy), clean = dataset[20]
    print(vframes.shape, noisy.shape, clean.shape)
