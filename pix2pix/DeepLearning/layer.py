import torch
import torch.nn as nn


class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        """
        * UNet encoding 모듈 구조 정의
        :param in_channels: in_channels 수
        :param out_channels: out_channels 수
        :param normalize: InstanceNorm2d 여부
        :param dropout: dropout 비율
        """

        super(UNetDown, self).__init__()

        # Conv2d
        layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        # InstanceNorm2d
        if normalize:
            layers.append(nn.InstanceNorm2d(num_features=out_channels))
        # LeakyReLU
        layers.append(nn.LeakyReLU(negative_slope=0.2))
        # Dropout
        if dropout:
            layers.append(nn.Dropout(p=dropout))

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        """
        * 순전파
        :param x: 배치 개수 만큼의 입력. (N, in_channels, H, W)
        :return: 배치 개수 만큼의 출력. (N, out_channels, H/2, W/2)
        """

        # (N, in_channels, H, W) -> (N, out_channels, H/2, W/2)
        x = self.down(x)

        return x


class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        """
        * UNet decoding 모듈 구조 정의
        :param in_channels: in_channels 수
        :param out_channels: out_channels 수
        :param dropout: dropout 비율
        """

        super(UNetUp, self).__init__()

        # ConvTranspose2d, InstanceNorm2d, LeakyReLU
        layers = [
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=1e-2)
        ]
        # Dropout
        if dropout:
            layers.append(nn.Dropout(p=dropout))

        self.up = nn.Sequential(*layers)

    def forward(self, x, skip):
        """
        * 순전파
        :param x: 배치 개수 만큼의 입력. (N, in_channels, H, W)
        :param skip: 배치 개수 만큼의 skip 입력. (N, skip.shape[1], H*2, W*2)
        :return: 배치 개수 만큼의 출력. (N, out_channels+skip.shape[1], H*2, W*2)
        """

        # (N, in_channels, H, W) -> (N, out_channels, H*2, W*2)
        x = self.up(x)
        # (N, out_channels, H*2, W*2) -> (N, out_channels+skip.shape[1], H*2, W*2)
        x = torch.cat(tensors=(x, skip), dim=1)

        return x


class DisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        """
        * DisBlock 모듈 구조 정의
        :param in_channels: in_channels 수
        :param out_channels: out_channels 수
        :param normalize: InstanceNorm2d 여부
        """

        super(DisBlock, self).__init__()

        # Conv2d
        layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=True)]
        # InstanceNorm2d
        if normalize:
            layers.append(nn.InstanceNorm2d(num_features=out_channels))
        # LeakyReLU
        layers.append(nn.LeakyReLU(negative_slope=0.2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """
        * 순전파
        :param x: 배치 개수 만큼의 입력. (N, in_channels, H, W)
        :return: 배치 개수 만큼의 출력. (N, out_channels, H/2, W/2)
        """

        # (N, in_channels, H, W) -> (N, out_channels, H/2, W/2)
        x = self.block(x)

        return x
