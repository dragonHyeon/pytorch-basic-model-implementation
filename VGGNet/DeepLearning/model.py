import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, cfg, in_channels, num_classes, init_weights):
        """
        * 모델 구조 정의
        :param cfg: VGG 모델 옵션 (VGGNet feature extractor 옵션)
        :param in_channels: in_channels 수
        :param num_classes: 출력 클래스 개수
        :param init_weights: 가중치 초기화 여부
        """

        super(VGG, self).__init__()

        # feature extractor
        self.features = make_layers(cfg=cfg, in_channels=in_channels)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

        # 가중치 초기화
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        """
        * 순전파
        :param x: 배치 개수 만큼의 입력. (N, in_channels (3), 224, 224)
        :return: 배치 개수 만큼의 출력. (N, num_classes (100))
        """

        # (N, in_channels (3), 224, 224) -> (N, 512, 7, 7)
        x = self.features(x)
        # (N, 512, 7, 7) -> (N, 512 * 7 * 7)
        x = torch.flatten(x, 1)
        # (N, 512 * 7 * 7) -> (N, num_classes (100))
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        """
        * 모델 가중치 초기화
        :return: 모델 가중치 초기화 진행됨
        """

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(tensor=m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(tensor=m.bias, val=0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(tensor=m.weight, val=1)
                nn.init.constant_(tensor=m.bias, val=0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(tensor=m.weight, mean=0, std=0.01)
                nn.init.constant_(tensor=m.bias, val=0)


def make_layers(cfg, in_channels):
    """
    * VGGNet feature extractor
    :param cfg: 'A', 'B', 'D', 'E' 중에 선택 (VGG-11, VGG-13, VGG-16, VGG-19)
    :param in_channels: in_channels 수
    :return: VGGNet feature extractor 만들어 줌
    """

    # VGGNet feature extractor 담을 리스트
    layers = []

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, stride=1, padding=1)]
            layers += [nn.ReLU()]
            in_channels = v

    return nn.Sequential(*layers)


# VGG 모델 옵션 (VGGNet feature extractor 옵션)
cfgs = {
    # VGG-11
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    # VGG-13
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    # VGG-16
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    # VGG-19
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def vgg11(in_channels=3, num_classes=100, init_weights=True):
    """
    * VGG-11
    :param in_channels: in_channels 수
    :param num_classes: 출력 클래스 개수
    :param init_weights: 가중치 초기화 여부
    :return: VGG 11-layer 모델
    """

    return VGG(cfg=cfgs['A'], in_channels=in_channels, num_classes=num_classes, init_weights=init_weights)


def vgg13(in_channels=3, num_classes=100, init_weights=True):
    """
    * VGG-13
    :param in_channels: in_channels 수
    :param num_classes: 출력 클래스 개수
    :param init_weights: 가중치 초기화 여부
    :return: VGG 13-layer 모델
    """

    return VGG(cfg=cfgs['B'], in_channels=in_channels, num_classes=num_classes, init_weights=init_weights)


def vgg16(in_channels=3, num_classes=100, init_weights=True):
    """
    * VGG-16
    :param in_channels: in_channels 수
    :param num_classes: 출력 클래스 개수
    :param init_weights: 가중치 초기화 여부
    :return: VGG 16-layer 모델
    """

    return VGG(cfg=cfgs['D'], in_channels=in_channels, num_classes=num_classes, init_weights=init_weights)


def vgg19(in_channels=3, num_classes=100, init_weights=True):
    """
    * VGG-19
    :param in_channels: in_channels 수
    :param num_classes: 출력 클래스 개수
    :param init_weights: 가중치 초기화 여부
    :return: VGG 19-layer 모델
    """

    return VGG(cfg=cfgs['E'], in_channels=in_channels, num_classes=num_classes, init_weights=init_weights)
