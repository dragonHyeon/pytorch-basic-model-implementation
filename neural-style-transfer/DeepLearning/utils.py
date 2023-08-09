import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from Lib import UtilLib, DragonLib
from Common import ConstVar


def load_checkpoint(filepath):
    """
    * 체크포인트 불러오기
    :param filepath: 불러올 체크포인트 파일 경로
    :return: state 모음 (model.state_dict(), optimizer.state_dict(), epoch, score)
    """

    # state 불러오기
    state = torch.load(f=filepath)

    # state 정보 리턴
    return state


def save_checkpoint(filepath, original_image, optimizer=None, epoch=None, score=None, is_best=False):
    """
    * 체크포인트 저장
    :param filepath: 저장될 체크포인트 파일 경로
    :param original_image: 저장될 원본 이미지
    :param optimizer: 저장될 optimizer
    :param epoch: 저장될 현재 학습 epoch 횟수
    :param score: 저장될 현재 score
    :param is_best: 현재 저장하려는 모델이 가장 좋은 성능의 모델인지 여부
    :return: 체크포인트 파일 생성됨
    """

    # 저장하고자 하는 경로의 상위 디렉터리가 존재하지 않는 경우 상위 경로 생성
    DragonLib.make_parent_dir_if_not_exits(target_path=filepath)

    # state 정보 담기
    state = {
        ConstVar.KEY_STATE_ORIGINAL_IMAGE: original_image,
        ConstVar.KEY_STATE_OPTIMIZER: optimizer.state_dict(),
        ConstVar.KEY_STATE_EPOCH: epoch,
        ConstVar.KEY_STATE_SCORE: score
    }

    # state 저장
    torch.save(obj=state,
               f=filepath)

    # 현재 저장하려는 모델이 가장 좋은 성능의 모델인 경우 best model 로 저장
    if is_best:
        torch.save(obj=state,
                   f=UtilLib.getNewPath(path=UtilLib.getParentDirPath(filePath=filepath),
                                        add=ConstVar.CHECKPOINT_BEST_FILE_NAME))


def get_gram_matrix(feature):
    """
    * feature 를 이용하여 gram matrix 계산
    :param feature: VGG-19 의 intermediate output value
    :return: gram matrix
    """

    # gram matrix 계산
    n, c, h, w = feature.size()
    feature = feature.view(n * c, h * w)
    gram_matrix = torch.mm(feature, feature.t())

    return gram_matrix


def read_img(filepath):
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
    img = torch.unsqueeze(input=transform(Image.open(fp=filepath)), dim=0)

    return img
