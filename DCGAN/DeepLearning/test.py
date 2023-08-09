import torch
from torchvision import transforms

from Common import ConstVar
from Lib import UtilLib, DragonLib


class Tester:
    def __init__(self, modelG, device):
        """
        * 테스트 관련 클래스
        :param modelG: 테스트 할 모델. 생성자
        :param device: GPU / CPU
        """

        # 테스트 할 모델
        self.modelG = modelG
        # GPU / CPU
        self.device = device

    def running(self, nums_to_generate, output_dir, generated_folder_name):
        """
        * 테스트 셋팅 및 진행
        :param nums_to_generate: 생성할 이미지 개수
        :param output_dir: 결과물 파일 저장할 디렉터리 위치
        :param generated_folder_name: 생성된 이미지 파일 저장될 폴더명
        :return: 테스트 수행됨
        """

        # 생성할 이미지 개수
        self.nums_to_generate = nums_to_generate

        # 테스트 진행
        fake_x_batch = self._test()

        # standardization 하는데 사용된 std, mean 값
        mean = torch.tensor(ConstVar.NORMALIZE_MEAN)
        std = torch.tensor(ConstVar.NORMALIZE_STD)

        for num, fake_x in enumerate(iterable=fake_x_batch, start=1):

            # 시각화를 위해 standardization 한 거 원래대로 되돌리기, 값 범위 0 에서 1 로 제한 및 PIL image 로 변환
            fake_x_pil = self._convert_img(fake_x=fake_x, mean=mean, std=std)

            # 결과물 이미지 저장
            generated_img_dir = UtilLib.getNewPath(path=output_dir, add=ConstVar.OUTPUT_DIR_SUFFIX_GENERATED_IMG.format(generated_folder_name))
            generated_img_filepath = UtilLib.getNewPath(path=generated_img_dir, add=ConstVar.GENERATED_IMG_FILE_NAME.format(num))
            self._save_pics(fake_x_pil=fake_x_pil, filepath=generated_img_filepath)

    def _test(self):
        """
        * 테스트 진행
        :return: 이미지 생성
        """

        # 모델을 테스트 모드로 전환
        self.modelG.eval()

        # 이미지 생성
        z = torch.randn(size=(self.nums_to_generate, 100, 1, 1), device=self.device)
        fake_x_batch = self.modelG(z)

        return fake_x_batch

    @staticmethod
    def _convert_img(fake_x, mean, std):
        """
        * normalize (혹은 standardize) 된 데이터를 원래 데이터로 되돌리고 값 범위 0 에서 1 사이로 제한해주며 PIL image 로 바꿔주기
        :param fake_x: 생성된 이미지
        :param mean: mean 값
        :param std: std 값
        :return: 변환된 형태의 PIL image
        """

        # tensor 에서 PIL 로 변환시켜주는 함수
        transform = transforms.ToPILImage()

        # 정규화된 데이터 원래 데이터로 돌려놓기
        fake_x = fake_x.cpu().detach() * std[:, None, None] + mean[:, None, None]

        # 값의 범위를 0 에서 1 로 제한
        fake_x[fake_x > 1] = 1
        fake_x[fake_x < 0] = 0

        # PIL image 로 변환
        fake_x_pil = transform(fake_x)

        return fake_x_pil

    @staticmethod
    def _save_pics(fake_x_pil, filepath):
        """
        * 이미지 파일 저장
        :param fake_x_pil: 생성된 PIL image 형식의 이미지
        :param filepath: 저장될 그림 파일 경로
        :return: 그림 파일 생성됨
        """

        # 저장하고자 하는 경로의 상위 디렉터리가 존재하지 않는 경우 상위 경로 생성
        DragonLib.make_parent_dir_if_not_exits(target_path=filepath)

        # 그림 저장
        fake_x_pil.save(fp=filepath)
