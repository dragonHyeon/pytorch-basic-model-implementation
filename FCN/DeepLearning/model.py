import torch.nn as nn
from torchvision.models import vgg16

# VGG-16 Max pooling 단위로 자르기
ranges = {'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31))}


class VGGNet(nn.Module):
    def __init__(self, pretrained=True):
        """
        * 모델 구조 정의
        * FCNs 의 backbone 네트워크로 사용될 네트워크 (VGG-16)
        :param pretrained: 미리 학습된 가중치 불러오기 여부
        """

        super(VGGNet, self).__init__()

        # VGG-16 Max pooling 단위로 자르기 위한 준비
        self.ranges = ranges['vgg16']

        # (N, 3, 224, 224) -> (N, 512, 7, 7)
        self.features = vgg16(pretrained=pretrained).features

    def forward(self, x):
        """
        * 순전파
        :param x: 배치 개수 만큼의 입력. (N, 3, 224, 224)
        :return: Max pooling 단위 결과 모음 dict
        """

        # Max pooling 단위로 끊어서 결과 담기 위한 dict
        output = {}

        # Max pooling 단위로 끊어서 결과 담기
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output['x{0}'.format(idx+1)] = x

        return output


class FCNs(nn.Module):
    def __init__(self, pretrained_net, num_classes):
        """
        * 모델 구조 정의
        :param pretrained_net: 미리 학습된 backbone 네트워크 (VGG-16)
        :param num_classes: 출력 클래스 개수 (image segmentation 시 분류해야 할 전체 label 의 개수)
        """

        super(FCNs, self).__init__()

        self.num_classes = num_classes

        self.relu = nn.ReLU(inplace=True)

        # feature extractor
        # (N, 3, 224, 224) -> (N, 512, 7, 7)
        self.pretrained_net = pretrained_net

        # (N, 512, 7, 7) -> (N, 512, 14, 14)
        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)
        self.bn1 = nn.BatchNorm2d(num_features=512)
        # (N, 512, 14, 14) -> (N, 256, 28, 28)
        self.deconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)
        self.bn2 = nn.BatchNorm2d(num_features=256)
        # (N, 256, 28, 28) -> (N, 128, 56, 56)
        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        # (N, 128, 56, 56) -> (N, 64, 112, 112)
        self.deconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)
        self.bn4 = nn.BatchNorm2d(num_features=64)
        # (N, 64, 112, 112) -> (N, 32, 224, 224)
        self.deconv5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)
        self.bn5 = nn.BatchNorm2d(num_features=32)

        # NIN
        # (N, 32, 224, 224) -> (N, num_classes (2), 224, 224)
        self.classifier = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        """
        * 순전파
        :param x: 배치 개수 만큼의 입력. (N, 3, 224, 224)
        :return: 배치 개수 만큼의 출력. (N, num_classes (2), 224, 224)
        """

        # Max pooling 단위 결과 모음 dict
        output = self.pretrained_net(x)

        # (N, 3, 224, 224) -> (N, 64, 112, 112)
        x1 = output['x1']
        # (N, 64, 112, 112) -> (N, 128, 56, 56)
        x2 = output['x2']
        # (N, 128, 56, 56) -> (N, 256, 28, 28)
        x3 = output['x3']
        # (N, 256, 28, 28) -> (N, 512, 14, 14)
        x4 = output['x4']
        # (N, 512, 14, 14) -> (N, 512, 7, 7)
        x5 = output['x5']

        # (N, 512, 7, 7) -> (N, 512, 14, 14)
        score = x4 + self.bn1(self.relu(self.deconv1(x5)))
        # (N, 512, 14, 14) -> (N, 256, 28, 28)
        score = x3 + self.bn2(self.relu(self.deconv2(score)))
        # (N, 256, 28, 28) -> (N, 128, 56, 56)
        score = x2 + self.bn3(self.relu(self.deconv3(score)))
        # (N, 128, 56, 56) -> (N, 64, 112, 112)
        score = x1 + self.bn4(self.relu(self.deconv4(score)))
        # (N, 64, 112, 112) -> (N, 32, 224, 224)
        score = self.bn5(self.relu(self.deconv5(score)))
        # (N, 32, 224, 224) -> (N, num_classes (2), 224, 224)
        score = self.classifier(score)

        return score
