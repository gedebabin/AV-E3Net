from torch.hub import load_state_dict_from_url
from torchvision.models.shufflenetv2 import ShuffleNetV2
from torch import Tensor


class ShuffleNetV2Encoder05(ShuffleNetV2):
    def __init__(self) -> None:
        super().__init__([4, 8, 4], [24, 48, 96, 192, 1024])
        delattr(self, 'fc')

    def _forward_impl(self, x: Tensor) -> Tensor:
        '''
            Override forward_impl to skip last fc layer. Return features 
        '''
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # globalpool
        # x = self.fc(x)
        # print('shufflenet output shape', x.shape)
        return x


def _shufflenetv2_05() -> ShuffleNetV2Encoder05:
    model = ShuffleNetV2Encoder05()
    model_url = "https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth"
    state_dict = load_state_dict_from_url(model_url, progress=True)
    del state_dict['fc.weight']
    del state_dict['fc.bias']
    model.load_state_dict(state_dict)
    return model
