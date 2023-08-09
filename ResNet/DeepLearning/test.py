import torch
from torchvision import transforms
from PIL import Image

from Common import ConstVar
from DeepLearning import utils


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

    def running(self, input_path):
        """
        * 테스트 셋팅 및 진행
        :param input_path: 입력 이미지 파일 경로
        :return: 테스트 수행됨
        """

        # 테스트 진행
        y_pred = self._test(input_path=input_path)

        # classification 결과물을 하나의 class 로 변환
        predicted_class = self._get_class(y_pred=y_pred)

        # classification 결과 출력
        print(predicted_class)

    def _test(self, input_path):
        """
        * 테스트 진행
        :param input_path: 입력 이미지 파일 경로
        :return: 이미지 classification 결과 반환
        """

        # 모델을 테스트 모드로 전환
        self.model.eval()

        # 이미지 읽고 변환하기
        img = self._read_img(filepath=input_path)
        img = img.unsqueeze(dim=0)

        # 이미지 classification 진행
        img = img.to(self.device)
        y_pred = self.model(img)

        return y_pred

    @staticmethod
    def _read_img(filepath):
        """
        * 이미지 읽고 변환하기
        :param filepath: 읽어 올 이미지 파일 경로
        :return: 이미지 읽어 변환 해줌
        """

        # 데이터 변환 함수
        transform = transforms.Compose([
            transforms.Resize(size=ConstVar.RESIZE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=ConstVar.NORMALIZE_MEAN, std=ConstVar.NORMALIZE_STD)
        ])

        # 이미지 읽기 및 변환
        img = transform(Image.open(fp=filepath))

        return img

    @staticmethod
    def _get_class(y_pred):
        """
        * classification 결과물을 하나의 class 로 변환
        :param y_pred: classification 결과물
        :return: 변환된 class 값
        """

        # classification 결과물의 idx 값 추출
        idx = torch.argmax(input=y_pred, dim=1).item()

        # idx 에 해당하는 class 가져오기
        predicted_class = utils.idx_to_class(idx=idx)

        return predicted_class
