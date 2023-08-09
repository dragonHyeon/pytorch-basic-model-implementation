import torch
import torch.nn as nn


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        """
        * Inception 모듈 구조 정의
        :param in_channels: in_channels 수
        :param ch1x1: branch1 conv1x1 out_channels 수
        :param ch3x3red: branch2 conv3x3 in_channels 수
        :param ch3x3: branch2 conv3x3 out_channels 수
        :param ch5x5red: branch3 conv5x5 in_channels 수
        :param ch5x5: branch3 conv5x5 out_channels 수
        :param pool_proj: branch4 conv1x1 out_channels 수
        """

        super(Inception, self).__init__()

        # conv1x1
        self.branch1 = nn.Sequential(
            # (N, in_channels, H, W) -> (N, ch1x1, H, W)
            BasicConv2d(in_channels=in_channels, out_channels=ch1x1, kernel_size=1, stride=1, padding=0)
        )

        # conv1x1 - conv3x3
        self.branch2 = nn.Sequential(
            # (N, in_channels, H, W) -> (N, ch3x3red, H, W)
            BasicConv2d(in_channels=in_channels, out_channels=ch3x3red, kernel_size=1, stride=1, padding=0),
            # (N, ch3x3red, H, W) -> (N, ch3x3, H, W)
            BasicConv2d(in_channels=ch3x3red, out_channels=ch3x3, kernel_size=3, stride=1, padding=1)
        )

        # conv1x1 - conv5x5
        self.branch3 = nn.Sequential(
            # (N, in_channels, H, W) -> (N, ch5x5red, H, W)
            BasicConv2d(in_channels=in_channels, out_channels=ch5x5red, kernel_size=1, stride=1, padding=0),
            # (N, ch5x5red, H, W) -> (N, ch5x5, H, W)
            BasicConv2d(in_channels=ch5x5red, out_channels=ch5x5, kernel_size=5, stride=1, padding=2)
        )

        # maxpool3x3 - conv1x1
        self.branch4 = nn.Sequential(
            # (N, in_channels, H, W) -> (N, in_channels, H, W)
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            # (N, in_channels, H, W) -> (N, pool_proj, H, W)
            BasicConv2d(in_channels=in_channels, out_channels=pool_proj, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        """
        * 순전파
        :param x: 배치 개수 만큼의 입력. (N, in_channels, H, W)
        :return: 배치 개수 만큼의 출력. (N, ch1x1+ch3x3+ch5x5+pool_proj, H, W)
        """

        # (N, in_channels, H, W) -> (N, ch1x1, H, W)
        branch1 = self.branch1(x)
        # (N, in_channels, H, W) -> (N, ch3x3, H, W)
        branch2 = self.branch2(x)
        # (N, in_channels, H, W) -> (N, ch5x5, H, W)
        branch3 = self.branch3(x)
        # (N, in_channels, H, W) -> (N, pool_proj, H, W)
        branch4 = self.branch4(x)

        # (N, in_channels, H, W) -> (N, ch1x1+ch3x3+ch5x5+pool_proj, H, W)
        out = torch.cat(tensors=[branch1, branch2, branch3, branch4], dim=1)

        return out


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes=100):
        """
        * InceptionAux 모듈 구조 정의
        :param in_channels: in_channels 수
        :param num_classes: 출력 클래스 개수
        """

        super(InceptionAux, self).__init__()

        # (N, in_channels, H, W) -> (N, in_channels, 4, 4)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(4, 4))
        # (N, in_channels, 4, 4) -> (N, 128, 4, 4)
        self.conv = BasicConv2d(in_channels=in_channels, out_channels=128, kernel_size=1, stride=1, padding=0)
        # (N, 2048) -> (N, 1024)
        self.fc1 = nn.Linear(in_features=2048, out_features=1024, bias=True)
        # (N, 1024) -> (N, num_classes (100))
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes, bias=True)

    def forward(self, x):
        """
        * 순전파
        :param x: 배치 개수 만큼의 입력. (N, in_channels, H, W)
        :return: 배치 개수 만큼의 출력. (N, num_classes (100))
        """

        # (N, in_channels, H, W) -> (N, in_channels, 4, 4)
        out = self.avgpool(x)
        # (N, in_channels, 4, 4) -> (N, 128, 4, 4)
        out = self.conv(out)
        # (N, 128, 4, 4) -> (N, 128 * 4 * 4)
        out = torch.flatten(out, 1)
        # (N, 2048) -> (N, 1024)
        out = self.fc1(out)
        # (N, 1024) -> (N, num_classes (100))
        out = self.fc2(out)

        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        * BasicConv2d 모듈 구조 정의
        :param in_channels: in_channels 수
        :param out_channels: out_channels 수
        :param kernel_size: kernel 사이즈
        :param stride: stride 값
        :param padding: padding 크기
        """

        super(BasicConv2d, self).__init__()

        # (Conv + BatchNorm + ReLU) 블록
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_channels, eps=0.001),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        * 순전파
        :param x: 배치 개수 만큼의 입력. (N, in_channels, H, W)
        :return: 배치 개수 만큼의 출력. (N, out_channels, (H-kernel_size+2*padding)/stride+1, (W-kernel_size+2*padding)/stride+1)
        """

        # (N, in_channels, H, W) -> (N, out_channels, (H-kernel_size+2*padding)/stride+1, (W-kernel_size+2*padding)/stride+1)
        out = self.block(x)

        return out
