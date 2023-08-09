import torch
import torch.nn as nn

from DeepLearning.layer import Inception, InceptionAux, BasicConv2d


class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=100, aux_logits=True, init_weights=True):
        """
        * 모델 구조 정의
        :param in_channels: in_channels 수
        :param num_classes: 출력 클래스 개수
        :param aux_logits: auxiliary classifier 여부
        :param init_weights: 가중치 초기화 여부
        """

        super(GoogLeNet, self).__init__()

        # auxiliary classifier 여부
        self.aux_logits = aux_logits

        # (N, in_channels (3), H (224), W (224)) -> (N, 64, 112, 112)
        self.conv1 = BasicConv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        # (N, 64, 112, 112) -> (N, 64, 56, 56)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        # (N, 64, 56, 56) -> (N, 64, 56, 56)
        self.conv2 = BasicConv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        # (N, 64, 56, 56) -> (N, 192, 56, 56)
        self.conv3 = BasicConv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        # (N, 192, 56, 56) -> (N, 192, 28, 28)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # (N, 192, 28, 28) -> (N, 256, 28, 28)
        self.inception3a = Inception(in_channels=192, ch1x1=64, ch3x3red=96, ch3x3=128, ch5x5red=16, ch5x5=32, pool_proj=32)
        # (N, 256, 28, 28) -> (N, 480, 28, 28)
        self.inception3b = Inception(in_channels=256, ch1x1=128, ch3x3red=128, ch3x3=192, ch5x5red=32, ch5x5=96, pool_proj=64)
        # (N, 480, 28, 28) -> (N, 480, 14, 14)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # (N, 480, 14, 14) -> (N, 512, 14, 14)
        self.inception4a = Inception(in_channels=480, ch1x1=192, ch3x3red=96, ch3x3=208, ch5x5red=16, ch5x5=48, pool_proj=64)
        # (N, 512, 14, 14) -> (N, 512, 14, 14)
        self.inception4b = Inception(in_channels=512, ch1x1=160, ch3x3red=112, ch3x3=224, ch5x5red=24, ch5x5=64, pool_proj=64)
        # (N, 512, 14, 14) -> (N, 512, 14, 14)
        self.inception4c = Inception(in_channels=512, ch1x1=128, ch3x3red=128, ch3x3=256, ch5x5red=24, ch5x5=64, pool_proj=64)
        # (N, 512, 14, 14) -> (N, 528, 14, 14)
        self.inception4d = Inception(in_channels=512, ch1x1=112, ch3x3red=144, ch3x3=288, ch5x5red=32, ch5x5=64, pool_proj=64)
        # (N, 528, 14, 14) -> (N, 832, 14, 14)
        self.inception4e = Inception(in_channels=528, ch1x1=256, ch3x3red=160, ch3x3=320, ch5x5red=32, ch5x5=128, pool_proj=128)
        # (N, 832, 14, 14) -> (N, 832, 7, 7)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        # (N, 832, 7, 7) -> (N, 832, 7, 7)
        self.inception5a = Inception(in_channels=832, ch1x1=256, ch3x3red=160, ch3x3=320, ch5x5red=32, ch5x5=128, pool_proj=128)
        # (N, 832, 7, 7) -> (N, 1024, 7, 7)
        self.inception5b = Inception(in_channels=832, ch1x1=384, ch3x3red=192, ch3x3=384, ch5x5red=48, ch5x5=128, pool_proj=128)

        # (N, 1024, 7, 7) -> (N, 1024, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(p=0.2)
        # (N, 1024) -> (N, num_classes (100))
        self.fc = nn.Linear(in_features=1024, out_features=num_classes, bias=True)

        # auxiliary classifier
        if self.aux_logits:
            # (N, 512, 14, 14) -> (N, num_classes (100))
            self.aux1 = InceptionAux(in_channels=512, num_classes=num_classes)
            # (N, 528, 14, 14) -> (N, num_classes (100))
            self.aux2 = InceptionAux(in_channels=528, num_classes=num_classes)

        # 가중치 초기화
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        """
        * 순전파
        :param x: 배치 개수 만큼의 입력. (N, in_channels (3), H (224), W (224))
        :return: 배치 개수 만큼의 출력. (N, num_classes (100))
        """

        # (N, in_channels (3), H (224), W (224)) -> (N, 64, 112, 112)
        out = self.conv1(x)
        # (N, 64, 112, 112) -> (N, 64, 56, 56)
        out = self.maxpool1(out)
        # (N, 64, 56, 56) -> (N, 64, 56, 56)
        out = self.conv2(out)
        # (N, 64, 56, 56) -> (N, 192, 56, 56)
        out = self.conv3(out)
        # (N, 192, 56, 56) -> (N, 192, 28, 28)
        out = self.maxpool2(out)

        # (N, 192, 28, 28) -> (N, 256, 28, 28)
        out = self.inception3a(out)
        # (N, 256, 28, 28) -> (N, 480, 28, 28)
        out = self.inception3b(out)
        # (N, 480, 28, 28) -> (N, 480, 14, 14)
        out = self.maxpool3(out)

        # (N, 480, 14, 14) -> (N, 512, 14, 14)
        out = self.inception4a(out)
        # auxiliary classifier 있는 경우 및 학습 시점
        if self.aux_logits and self.training:
            # (N, 512, 14, 14) -> (N, num_classes (100))
            aux1 = self.aux1(out)

        # (N, 512, 14, 14) -> (N, 512, 14, 14)
        out = self.inception4b(out)
        # (N, 512, 14, 14) -> (N, 512, 14, 14)
        out = self.inception4c(out)
        # (N, 512, 14, 14) -> (N, 528, 14, 14)
        out = self.inception4d(out)
        # auxiliary classifier 있는 경우 및 학습 시점
        if self.aux_logits and self.training:
            # (N, 528, 14, 14) -> (N, num_classes (100))
            aux2 = self.aux2(out)

        # (N, 528, 14, 14) -> (N, 832, 14, 14)
        out = self.inception4e(out)
        # (N, 832, 14, 14) -> (N, 832, 7, 7)
        out = self.maxpool4(out)

        # (N, 832, 7, 7) -> (N, 832, 7, 7)
        out = self.inception5a(out)
        # (N, 832, 7, 7) -> (N, 1024, 7, 7)
        out = self.inception5b(out)

        # (N, 1024, 7, 7) -> (N, 1024, 1, 1)
        out = self.avgpool(out)
        # (N, 1024, 1, 1) -> (N, 1024 * 1 * 1)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        # (N, 1024) -> (N, num_classes (100))
        out = self.fc(out)

        # auxiliary classifier 있는 경우 및 학습 시점
        if self.aux_logits and self.training:
            return aux1, aux2, out
        # auxiliary classifier 없는 경우 혹은 테스트 시점
        else:
            return out

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
