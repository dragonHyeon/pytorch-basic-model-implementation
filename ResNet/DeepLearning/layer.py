import torch.nn as nn


def conv3x3(in_channels, out_channels, stride):
    """
    * 3x3 convolution
    """

    # (N, in_channels, H, W) -> (N, out_channels, H/stride, W/stride)
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_channels, out_channels, stride):
    """
    * 1x1 convolution
    """

    # (N, in_channels, H, W) -> (N, out_channels, H/stride, W/stride)
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=False)


class BasicBlock(nn.Module):

    # 해당 residual block 의 출력 채널을 기존 out_channels 수 보다 몇 배 늘릴지
    expansion = 1

    def __init__(self, in_channels, out_channels, stride):
        """
        * BasicBlock 모듈 구조 정의
        :param in_channels: in_channels 수
        :param out_channels: out_channels 수
        :param stride: stride 값
        """

        super(BasicBlock, self).__init__()

        # conv3x3 - conv3x3
        self.layer = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            conv3x3(in_channels=in_channels, out_channels=out_channels, stride=stride),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            conv3x3(in_channels=out_channels, out_channels=out_channels * BasicBlock.expansion, stride=1)
        )

        # shortcut 에서 stride 혹은 channel 에 변화 줘야하는지 조건
        non_identity_condition1 = stride != 1
        non_identity_condition2 = in_channels != out_channels * BasicBlock.expansion

        # convolution block
        if non_identity_condition1 or non_identity_condition2:
            self.shortcut = conv1x1(in_channels=in_channels, out_channels=out_channels * BasicBlock.expansion, stride=stride)
        # identity block
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        """
        * 순전파
        :param x: 배치 개수 만큼의 입력. (N, in_channels, H, W)
        :return: 배치 개수 만큼의 출력. (N, out_channels*BottleNeck.expansion, H/stride, W/stride)
        """

        # (N, in_channels, H, W) -> (N, out_channels*BottleNeck.expansion, H/stride, W/stride)
        identity = self.shortcut(x)
        # (N, in_channels, H, W) -> (N, out_channels*BottleNeck.expansion, H/stride, W/stride)
        out = identity + self.layer(x)

        return out


class BottleNeck(nn.Module):

    # 해당 residual block 의 출력 채널을 기존 out_channels 수 보다 몇 배 늘릴지
    expansion = 4

    def __init__(self, in_channels, out_channels, stride):
        """
        * BottleNeck 모듈 구조 정의
        :param in_channels: in_channels 수
        :param out_channels: out_channels 수
        :param stride: stride 값
        """

        super(BottleNeck, self).__init__()

        # conv1x1 - conv3x3 - conv1x1
        self.layer = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            conv1x1(in_channels=in_channels, out_channels=out_channels, stride=stride),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            conv3x3(in_channels=out_channels, out_channels=out_channels, stride=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            conv1x1(in_channels=out_channels, out_channels=out_channels * BottleNeck.expansion, stride=1),
        )

        # shortcut 에서 stride 혹은 channel 에 변화 줘야하는지 조건
        non_identity_condition1 = stride != 1
        non_identity_condition2 = in_channels != out_channels * BottleNeck.expansion

        # convolution block
        if non_identity_condition1 or non_identity_condition2:
            self.shortcut = conv1x1(in_channels=in_channels, out_channels=out_channels * BottleNeck.expansion, stride=stride)
        # identity block
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        """
        * 순전파
        :param x: 배치 개수 만큼의 입력. (N, in_channels, H, W)
        :return: 배치 개수 만큼의 출력. (N, out_channels*BottleNeck.expansion, H/stride, W/stride)
        """

        # (N, in_channels, H, W) -> (N, out_channels*BottleNeck.expansion, H/stride, W/stride)
        identity = self.shortcut(x)
        # (N, in_channels, H, W) -> (N, out_channels*BottleNeck.expansion, H/stride, W/stride)
        out = identity + self.layer(x)

        return out
