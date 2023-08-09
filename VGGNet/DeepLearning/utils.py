import torch
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


def save_checkpoint(filepath, model, optimizer=None, epoch=None, score=None, is_best=False):
    """
    * 체크포인트 저장
    :param filepath: 저장될 체크포인트 파일 경로
    :param model: 저장될 모델
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
        ConstVar.KEY_STATE_MODEL: model.state_dict(),
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


def save_pics(pics_list, filepath, title):
    """
    * 학습 결과물 이미지로 저장
    :param pics_list: 원본, 재구성 이미지 쌍 담은 리스트
    :param filepath: 저장될 그림 파일 경로
    :param title: 그림 제목
    :return: 그림 파일 생성됨
    """

    # plt 로 시각화 할 수 있는 형식으로 변환
    plt_pics_list = [(original_x.cpu().reshape(-1, 32, 32).permute(1, 2, 0), reconstructed_x.cpu().detach().reshape(-1, 32, 32).permute(1, 2, 0)) for original_x, reconstructed_x in pics_list]

    # plt 에 그리기
    fig, axs = plt.subplots(nrows=10, ncols=2, figsize=(5, 15))
    fig.suptitle(t=title, fontsize=18)
    for num, (original_x, reconstructed_x) in enumerate(plt_pics_list):
        axs[num, 0].imshow(X=original_x, cmap='gray')
        axs[num, 0].axis('off')
        axs[num, 1].imshow(X=reconstructed_x, cmap='gray')
        axs[num, 1].axis('off')

    # 저장하고자 하는 경로의 상위 디렉터리가 존재하지 않는 경우 상위 경로 생성
    DragonLib.make_parent_dir_if_not_exits(target_path=filepath)

    # 그림 저장
    plt.savefig(filepath)


def idx_to_class(idx):
    """
    * idx 에 해당하는 class 가져오기
    :param idx: classification 결과물 idx
    :return: idx 에 해당하는 class 반환
    """

    # CIFAR-100 데이터셋 class 모음
    class_list = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                  'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
                  'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
                  'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo',
                  'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree',
                  'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree',
                  'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit',
                  'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper',
                  'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
                  'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe',
                  'whale', 'willow_tree', 'wolf', 'woman', 'worm']

    return class_list[idx]
