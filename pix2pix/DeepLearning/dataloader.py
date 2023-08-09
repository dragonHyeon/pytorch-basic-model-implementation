import os

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from Lib import UtilLib
from Common import ConstVar


class FACADESDataset(Dataset):
    def __init__(self, data_dir, direction, mode_train_test):
        """
        * FACADESDataset 데이터로더
        :param data_dir: 데이터 디렉터리
        :param direction: 어디서 어디로의 변환인지 선택
        :param mode_train_test: 학습 / 테스트 모드
        """

        # 데이터 해당 디렉터리
        self.data_dir = data_dir
        # a2b, b2a 변환 방향 선택
        self.direction = direction
        # 학습 / 테스트 모드
        self.mode_train_test = mode_train_test

        # a 데이터 디렉터리
        self.dir_a = UtilLib.getNewPath(path=self.data_dir,
                                        add='a')
        # b 데이터 디렉터리
        self.dir_b = UtilLib.getNewPath(path=self.data_dir,
                                        add='b')

        # a 파일 경로 모음
        self.files_a = [UtilLib.getNewPath(path=self.dir_a,
                                           add=filename)
                        for filename in os.listdir(self.dir_a)]
        # b 파일 경로 모음
        self.files_b = [UtilLib.getNewPath(path=self.dir_b,
                                           add=filename)
                        for filename in os.listdir(self.dir_b)]

        # 모드에 따른 데이터 전처리 방법
        self.transform = {
            ConstVar.MODE_TRAIN: transforms.Compose([
                transforms.Resize(size=(ConstVar.RESIZE_SIZE, ConstVar.RESIZE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=ConstVar.NORMALIZE_MEAN, std=ConstVar.NORMALIZE_STD)
            ]),
            ConstVar.MODE_TEST: transforms.Compose([
                transforms.Resize(size=(ConstVar.RESIZE_SIZE, ConstVar.RESIZE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=ConstVar.NORMALIZE_MEAN, std=ConstVar.NORMALIZE_STD)
            ])
        }

    def __len__(self):
        return len(self.files_a)

    def __getitem__(self, item):
        # a 데이터
        a = self.transform[self.mode_train_test](Image.open(fp=self.files_a[item]))
        # b 데이터
        b = self.transform[self.mode_train_test](Image.open(fp=self.files_b[item]))

        # a 에서 b 로의 변환
        if self.direction == ConstVar.A2B:
            return a, b
        # b 에서 a 로의 변환
        elif self.direction == ConstVar.B2A:
            return b, a
