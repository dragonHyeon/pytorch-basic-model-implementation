import torch.nn as nn
from torchvision.models import vgg19

# VGG-19 필요한 Conv 단위로 자르기
ranges = {'vgg19': ((0, 1), (1, 6), (6, 11), (11, 20), (20, 29))}


class VGGNet(nn.Module):
    def __init__(self):
        """
        * 모델 구조 정의
        * 이미지 특징 추출용으로 사용될 네트워크 (VGG-19)
        """

        super(VGGNet, self).__init__()

        # VGG-19 필요한 Conv 단위로 자르기 위한 준비
        self.ranges = ranges['vgg19']

        # (N, 3, H, W) -> (N, 512, 7, 7)
        self.features = vgg19(pretrained=True).features

    def forward(self, x):
        """
        * 순전파
        :param x: 배치 개수 만큼의 입력. (N, 3, 224, 224)
        :return: 필요한 Conv 단위 결과 모음 dict
        """

        # 필요한 Conv 단위로 끊어서 결과 담기 위한 dict
        output = {}

        # 필요한 Conv 단위로 끊어서 결과 담기
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output['conv{0}_1'.format(idx+1)] = x

        return output
