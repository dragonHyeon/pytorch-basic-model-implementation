import os

import torch
from torchvision import transforms
from PIL import Image

from Common import ConstVar
from Lib import UtilLib, DragonLib


class Tester:
    def __init__(self, model, device):
        """
        * 테스트 관련 클래스
        :param model: 테스트 할 모델
        :param device: GPU / CPU
        """

        # 테스트 할 모델
        self.model = model
        # GPU / CPU
        self.device = device

    def running(self, input_dir, output_dir, segmentation_folder_name):
        """
        * 테스트 셋팅 및 진행
        :param input_dir: 입력 이미지 파일 디렉터리 위치
        :param output_dir: 결과물 파일 저장할 디렉터리 위치
        :param segmentation_folder_name: segmentation 된 이미지 파일 저장될 폴더명
        :return: 테스트 수행됨
        """

        # 테스트 진행
        result_list = self._test(input_dir=input_dir)

        for y_pred, img_filepath in result_list:

            # 시각화를 위해 prediction map 으로 변환 및 PIL image 로 변환
            prediction_map_pil = self._convert_data(y_pred=y_pred)

            # 이미지 저장 경로
            img_filename = UtilLib.getOnlyFileName(filePath=img_filepath)
            segmentation_img_dir = UtilLib.getNewPath(path=output_dir, add=ConstVar.OUTPUT_DIR_SUFFIX_SEGMENTATION_IMG.format(segmentation_folder_name))
            original_img_dir = UtilLib.getNewPath(path=segmentation_img_dir, add=ConstVar.OUTPUT_DIR_SUFFIX_ORIGINAL)
            segmented_img_dir = UtilLib.getNewPath(path=segmentation_img_dir, add=ConstVar.OUTPUT_DIR_SUFFIX_SEGMENTED)
            original_img_filepath = UtilLib.getNewPath(path=original_img_dir, add=ConstVar.SEGMENTATION_IMG_FILE_NAME.format(img_filename))
            segmented_img_filepath = UtilLib.getNewPath(path=segmented_img_dir, add=ConstVar.SEGMENTATION_IMG_FILE_NAME.format(img_filename))

            # 원본 이미지 저장
            DragonLib.copyfile_with_make_parent_dir(src=img_filepath, dst=original_img_filepath)
            # 결과물 이미지 저장
            self._save_pics(prediction_map_pil=prediction_map_pil, filepath=segmented_img_filepath)

    def _test(self, input_dir):
        """
        * 테스트 진행
        :param input_dir: 입력 이미지 파일 디렉터리 위치
        :return: 이미지 segmentation 결과물 및 파일 경로 반환
        """

        # segmentation 된 결과물 및 원본 이미지 경로명 담을 리스트
        result_list = []

        # 모델을 테스트 모드로 전환
        self.model.eval()

        for image_filename in os.listdir(input_dir):

            # 이미지 읽고 변환하기
            img_filepath = UtilLib.getNewPath(path=input_dir, add=image_filename)
            img = self._read_img(filepath=img_filepath)
            img = img.unsqueeze(dim=0)

            # 이미지 segmentation 진행
            img = img.to(self.device)
            y_pred = self.model(img)

            # segmentation 된 결과물과 원본 이미지 경로명 담기
            result_list.append((y_pred, img_filepath))

        return result_list

    @staticmethod
    def _read_img(filepath):
        """
        * 이미지 읽고 변환하기
        :param filepath: 읽어 올 이미지 파일 경로
        :return: 이미지 읽어 변환 해줌
        """

        # 데이터 변환 함수
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(size=(ConstVar.RESIZE_SIZE, ConstVar.RESIZE_SIZE)),
            transforms.ToTensor()
        ])

        # 이미지 읽기 및 변환
        img = transform(Image.open(fp=filepath))

        return img

    @staticmethod
    def _convert_data(y_pred):
        """
        * 시각화를 위해 segmentation 한 결과를 prediction map 으로 변환 및 PIL image 로 바꿔주기
        :param y_pred: segmentation 한 결과물
        :return: 변환된 형태의 PIL image
        """

        # tensor 에서 PIL 로 변환시켜주는 함수
        transform = transforms.ToPILImage()

        # segmentation 한 결과를 prediction map 으로 변환
        prediction_map = torch.argmax(input=y_pred, dim=1).type(torch.float16)

        # PIL image 로 변환
        prediction_map_pil = transform(prediction_map)

        return prediction_map_pil

    @staticmethod
    def _save_pics(prediction_map_pil, filepath):
        """
        * 이미지 파일 저장
        :param prediction_map_pil: segmentation 된 PIL image 형식의 이미지
        :param filepath: 저장될 그림 파일 경로
        :return: 그림 파일 생성됨
        """

        # 저장하고자 하는 경로의 상위 디렉터리가 존재하지 않는 경우 상위 경로 생성
        DragonLib.make_parent_dir_if_not_exits(target_path=filepath)

        # 그림 저장
        prediction_map_pil.save(fp=filepath)
