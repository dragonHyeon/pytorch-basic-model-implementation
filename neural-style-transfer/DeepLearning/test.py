import torch
from torchvision import transforms

from Common import ConstVar
from Lib import UtilLib, DragonLib


class Tester:
    def __init__(self, transferred_image):
        """
        * 테스트 관련 클래스
        :param transferred_image: 스타일 변환된 이미지
        """

        # 스타일 변환된 이미지
        self.transferred_image = transferred_image

    def running(self, output_dir, generated_folder_name):
        """
        * 테스트 셋팅 및 진행
        :param output_dir: 결과물 파일 저장할 디렉터리 위치
        :param generated_folder_name: 생성된 이미지 파일 저장될 폴더명
        :return: 테스트 수행됨
        """

        # standardization 하는데 사용된 std, mean 값
        mean = torch.tensor(ConstVar.NORMALIZE_MEAN)
        std = torch.tensor(ConstVar.NORMALIZE_STD)

        # 시각화를 위해 standardization 한 거 원래대로 되돌리기, 값 범위 0 에서 1 로 제한 및 PIL image 로 변환
        self._convert_img(mean=mean, std=std)

        # 결과물 이미지 저장
        generated_img_dir = UtilLib.getNewPath(path=output_dir, add=ConstVar.OUTPUT_DIR_SUFFIX_GENERATED_IMG.format(generated_folder_name))
        generated_img_filepath = UtilLib.getNewPath(path=generated_img_dir, add=ConstVar.GENERATED_IMG_FILE_NAME)
        self._save_pics(filepath=generated_img_filepath)

    def _convert_img(self, mean, std):
        """
        * normalize (혹은 standardize) 된 데이터를 원래 데이터로 되돌리고 값 범위 0 에서 1 사이로 제한해주며 PIL image 로 바꿔주기
        :param mean: mean 값
        :param std: std 값
        :return: 변환된 형태의 PIL image
        """

        # tensor 에서 PIL 로 변환시켜주는 함수
        transform = transforms.ToPILImage()

        # 정규화된 데이터 원래 데이터로 돌려놓기
        self.transferred_image = self.transferred_image[0].cpu().detach() * std[:, None, None] + mean[:, None, None]

        # 값의 범위를 0 에서 1 로 제한
        self.transferred_image[self.transferred_image > 1] = 1
        self.transferred_image[self.transferred_image < 0] = 0

        # PIL image 로 변환
        self.transferred_image = transform(self.transferred_image)

    def _save_pics(self, filepath):
        """
        * 이미지 파일 저장
        :param filepath: 저장될 그림 파일 경로
        :return: 그림 파일 생성됨
        """

        # 저장하고자 하는 경로의 상위 디렉터리가 존재하지 않는 경우 상위 경로 생성
        DragonLib.make_parent_dir_if_not_exits(target_path=filepath)

        # 그림 저장
        self.transferred_image.save(fp=filepath)
