import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from Lib import UtilLib
from Common import ConstVar


class CXRDataset(Dataset):
    def __init__(self, data_dir_x, data_dir_y, mode_train_test):
        """
        * CXRDataset 데이터로더
        :param data_dir_x: x 데이터 디렉터리
        :param data_dir_y: y 데이터 디렉터리
        :param mode_train_test: 학습 / 테스트 모드
        """

        # x 데이터 해당 디렉터리
        self.data_dir_x = data_dir_x
        # y 데이터 해당 디렉터리
        self.data_dir_y = data_dir_y
        # 학습 / 테스트 모드
        self.mode_train_test = mode_train_test

        # 모드에 따른 & x, y 에 따른 데이터 전처리 방법
        self.transform = {
            ConstVar.MODE_TRAIN: {
                ConstVar.KEY_X: transforms.Compose([
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize(size=(ConstVar.RESIZE_SIZE, ConstVar.RESIZE_SIZE)),
                    transforms.ToTensor()
                ]),
                ConstVar.KEY_Y: transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize(size=(ConstVar.RESIZE_SIZE, ConstVar.RESIZE_SIZE)),
                    transforms.ToTensor()
                ])
            },
            ConstVar.MODE_TEST: {
                ConstVar.KEY_X: transforms.Compose([
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize(size=(ConstVar.RESIZE_SIZE, ConstVar.RESIZE_SIZE)),
                    transforms.ToTensor()
                ]),
                ConstVar.KEY_Y: transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize(size=(ConstVar.RESIZE_SIZE, ConstVar.RESIZE_SIZE)),
                    transforms.ToTensor()
                ])
            },
        }

        # x, y 데이터 누락 없이 쌍 이루는 데이터만 고르기
        # self.x_path_list, self.y_path_list 생성됨
        self._pre_process()

    def __len__(self):
        return len(self.x_path_list)

    def __getitem__(self, item):
        # input
        x = self.transform[self.mode_train_test][ConstVar.KEY_X](Image.open(fp=self.x_path_list[item]))
        # ground truth
        y = self._mask_one_hot_encoding(
            mask=self.transform[self.mode_train_test][ConstVar.KEY_Y](Image.open(fp=self.y_path_list[item]))
        )

        return x, y

    def _pre_process(self):
        """
        * x, y 데이터 누락 없이 쌍 이루는 데이터만 고르기
        :return: self.x_path_list, self.y_path_list 만들어짐
        """

        # x 파일들의 파일명에서 주요 이름 모으기
        # 파일명 예시: 'CHNCXR_0013_0.jpg'
        # 주요 이름 예시: CHNCXR_0013_0
        x_core_filename_list = [UtilLib.getOnlyFileName(filePath=x_filename) for x_filename in os.listdir(self.data_dir_x)]

        # y 파일들의 파일명에서 주요 이름 모으기
        # 파일명 예시: 'CHNCXR_0013_0_mask.jpg'
        # 주요 이름 예시: CHNCXR_0013_0
        y_core_filename_list = [UtilLib.getOnlyFileName(filePath=y_filename)[:-5] for y_filename in os.listdir(self.data_dir_y)]

        # x, y 주요 이름 모음에서 공통되는 것만 모으기
        total_overlapping_core_filename_list = list(
            set(x_core_filename_list) & set(y_core_filename_list)
        )

        # 실제 사용될 x 데이터 경로 모음
        self.x_path_list = sorted([UtilLib.getNewPath(path=self.data_dir_x,
                                                      add='{0}.png'.format(core_filename)) for core_filename in total_overlapping_core_filename_list])

        # 실제 사용될 y 데이터 경로 모음
        self.y_path_list = sorted([UtilLib.getNewPath(path=self.data_dir_y,
                                                      add='{0}_mask.png'.format(core_filename)) for core_filename in total_overlapping_core_filename_list])

    @staticmethod
    def _mask_one_hot_encoding(mask):
        """
        * 원본 mask 데이터에 대하여 semantic segmentation 의 ground truth 형태로 one-hot encoding 및 변환 해주는 함수
        :param mask: 원본 mask 데이터. (1, 224, 224)
        :return: one-hot encoding 및 변환된 mask (semantic segmentation 의 ground truth 형태). (224, 224)
        """

        # one-hot encoding 및 변환된 mask 담기 위한 변수
        one_hot_mask = np.empty_like(mask, dtype=np.int64)

        # 배경에 해당하는 조건
        back_condition = (mask == 0)
        # 사물에 해당하는 조건
        object_condition = (mask > 0)

        # 클래스 값 부여. 값은 0, 1, 2, ... , num_class-1 로 이루어져 있을 거임. 이 경우는 클래스가 2 개라 0 과 1 만 존재
        # 배경 영역에 대해서는 0 부여
        one_hot_mask[back_condition] = 0
        # 사물 영역에 대해서는 1 부여
        one_hot_mask[object_condition] = 1

        # (1, 224, 224) -> (224, 224) 로 shape 바꾸기
        one_hot_mask = np.squeeze(a=one_hot_mask, axis=0)

        return one_hot_mask
