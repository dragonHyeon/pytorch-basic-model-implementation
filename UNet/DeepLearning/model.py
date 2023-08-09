import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        """
        * 모델 구조 정의
        :param in_channels: in_channels 수
        :param num_classes: 출력 클래스 개수
        """

        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            """
            * (Conv + BatchNorm + ReLU) 블록
            """

            # CBR 블록 담을 리스트
            layers = []

            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting Path
        # (N, in_channels (3), H, W) -> (N, 64, H, W)
        self.enc1_1 = CBR2d(in_channels=in_channels, out_channels=64)
        # (N, 64, H, W) -> (N, 64, H, W)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)
        # (N, 64, H, W) -> (N, 64, H/2, W/2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # (N, 64, H/2, W/2) -> (N, 128, H/2, W/2)
        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        # (N, 128, H/2, W/2) -> (N, 128, H/2, W/2)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)
        # (N, 128, H/2, W/2) -> (N, 128, H/4, W/4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # (N, 128, H/4, W/4) -> (N, 256, H/4, W/4)
        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        # (N, 256, H/4, W/4) -> (N, 256, H/4, W/4)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)
        # (N, 256, H/4, W/4) -> (N, 256, H/8, W/8)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # (N, 256, H/8, W/8) -> (N, 512, H/8, W/8)
        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        # (N, 512, H/8, W/8) -> (N, 512, H/8, W/8)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)
        # (N, 512, H/8, W/8) -> (N, 512, H/16, W/16)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # (N, 512, H/16, W/16) -> (N, 1024, H/16, W/16)
        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        # Expansive Path
        # (N, 1024, H/16, W/16) -> (N, 512, H/16, W/16)
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        # (N, 512, H/16, W/16) -> (N, 512, H/8, W/8)
        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2, padding=0, bias=True)
        # (N, 1024, H/8, W/8) -> (N, 512, H/8, W/8)
        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        # (N, 512, H/8, W/8) -> (N, 256, H/8, W/8)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        # (N, 256, H/8, W/8) -> (N, 256, H/4, W/4)
        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=True)
        # (N, 512, H/4, W/4) -> (N, 256, H/4, W/4)
        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        # (N, 256, H/4, W/4) -> (N, 128, H/4, W/4)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        # (N, 128, H/4, W/4) -> (N, 128, H/2, W/2)
        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0, bias=True)
        # (N, 256, H/2, W/2) -> (N, 128, H/2, W/2)
        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        # (N, 128, H/2, W/2) -> (N, 64, H/2, W/2)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        # (N, 64, H/2, W/2) -> (N, 64, H, W)
        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True)
        # (N, 128, H, W) -> (N, 64, H, W)
        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        # (N, 64, H, W) -> (N, 64, H, W)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        # NIN
        # (N, 64, H, W) -> (N, num_classes (2), H, W)
        self.fc = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        """
        * 순전파
        :param x: 배치 개수 만큼의 입력. (N, in_channels (3), H, W)
        :return: 배치 개수 만큼의 출력. (N, num_classes (2), H, W)
        """

        # (N, in_channels (3), H, W) -> (N, 64, H, W)
        enc1_1 = self.enc1_1(x)
        # (N, 64, H, W) -> (N, 64, H, W)
        enc1_2 = self.enc1_2(enc1_1)
        # (N, 64, H, W) -> (N, 64, H/2, W/2)
        pool1 = self.pool1(enc1_2)

        # (N, 64, H/2, H/2) -> (N, 128, H/2, W/2)
        enc2_1 = self.enc2_1(pool1)
        # (N, 128, H/2, W/2) -> (N, 128, H/2, W/2)
        enc2_2 = self.enc2_2(enc2_1)
        # (N, 128, H/2, W/2) -> (N, 128, H/4, W/4)
        pool2 = self.pool2(enc2_2)

        # (N, 128, H/4, W/4) -> (N, 256, H/4, W/4)
        enc3_1 = self.enc3_1(pool2)
        # (N, 256, H/4, W/4) -> (N, 256, H/4, W/4)
        enc3_2 = self.enc3_2(enc3_1)
        # (N, 256, H/4, W/4) -> (N, 256, H/8, W/8)
        pool3 = self.pool3(enc3_2)

        # (N, 256, H/8, W/8) -> (N, 512, H/8, W/8)
        enc4_1 = self.enc4_1(pool3)
        # (N, 512, H/8, W/8) -> (N, 512, H/8, W/8)
        enc4_2 = self.enc4_2(enc4_1)
        # (N, 512, H/8, W/8) -> (N, 512, H/16, W/16)
        pool4 = self.pool4(enc4_2)

        # (N, 512, H/16, W/16) -> (N, 1024, H/16, W/16)
        enc5_1 = self.enc5_1(pool4)

        # (N, 1024, H/16, W/16) -> (N, 512, H/16, W/16)
        dec5_1 = self.dec5_1(enc5_1)

        # (N, 512, H/16, W/16) -> (N, 512, H/8, W/8)
        unpool4 = self.unpool4(dec5_1)
        # unpool4 (N, 512, H/8, W/8) 에 enc4_2 (N, 512, H/8, W/8) concatenate 하여 cat4 (N, 1024, H/8, W/8) 만들기
        cat4 = torch.cat(tensors=(unpool4, enc4_2), dim=1)
        # (N, 1024, H/8, W/8) -> (N, 512, H/8, W/8)
        dec4_2 = self.dec4_2(cat4)
        # (N, 512, H/8, W/8) -> (N, 256, H/8, W/8)
        dec4_1 = self.dec4_1(dec4_2)

        # (N, 256, H/8, W/8) -> (N, 256, H/4, W/4)
        unpool3 = self.unpool3(dec4_1)
        # unpool3 (N, 256, H/4, W/4) 에 enc3_2 (N, 256, H/4, W/4) concatenate 하여 cat3 (N, 512, H/4, W/4) 만들기
        cat3 = torch.cat(tensors=(unpool3, enc3_2), dim=1)
        # (N, 512, H/4, W/4) -> (N, 256, H/4, W/4)
        dec3_2 = self.dec3_2(cat3)
        # (N, 256, H/4, W/4) -> (N, 128, H/4, W/4)
        dec3_1 = self.dec3_1(dec3_2)

        # (N, 128, H/4, W/4) -> (N, 128, H/2, W/2)
        unpool2 = self.unpool2(dec3_1)
        # unpool2 (N, 128, H/2, W/2) 에 enc2_2 (N, 128, H/2, W/2) concatenate 하여 cat2 (N, 256, H/2, W/2) 만들기
        cat2 = torch.cat(tensors=(unpool2, enc2_2), dim=1)
        # (N, 256, H/2, W/2) -> (N, 128, H/2, W/2)
        dec2_2 = self.dec2_2(cat2)
        # (N, 128, H/2, W/2) -> (N, 64, H/2, W/2)
        dec2_1 = self.dec2_1(dec2_2)

        # (N, 64, H/2, W/2) -> (N, 64, H, W)
        unpool1 = self.unpool1(dec2_1)
        # unpool1 (N, 64, H, W) 에 enc1_2 (N, 64, H, W) concatenate 하여 cat1 (N, 128, H, W) 만들기
        cat1 = torch.cat(tensors=(unpool1, enc1_2), dim=1)
        # (N, 128, H, W) -> (N, 64, H, W)
        dec1_2 = self.dec1_2(cat1)
        # (N, 64, H, W) -> (N, 64, H, W)
        dec1_1 = self.dec1_1(dec1_2)

        # (N, 64, H, W) -> (N, num_classes (2), H, W)
        out = self.fc(dec1_1)

        return out
