import torch
import matplotlib.pyplot as plt

from Lib import UtilLib, DragonLib
from Common import ConstVar


def load_checkpoint(filepath):
    """
    * 체크포인트 불러오기
    :param filepath: 불러올 체크포인트 파일 경로
    :return: state 모음 (model.state_dict(), optimizer.state_dict(), epoch)
    """

    # state 불러오기
    state = torch.load(f=filepath)

    # state 정보 리턴
    return state


def save_checkpoint(filepath, modelG, modelD, optimizerG, optimizerD, epoch, is_best=False):
    """
    * 체크포인트 저장
    :param filepath: 저장될 체크포인트 파일 경로
    :param modelG: 저장될 모델. 생성자
    :param modelD: 저장될 모델. 판별자
    :param optimizerG: 저장될 optimizer. 생성자
    :param optimizerD: 저장될 optimizer. 판별자
    :param epoch: 저장될 현재 학습 epoch 횟수
    :param is_best: 현재 저장하려는 모델이 가장 좋은 성능의 모델인지 여부
    :return: 체크포인트 파일 생성됨
    """

    # 저장하고자 하는 경로의 상위 디렉터리가 존재하지 않는 경우 상위 경로 생성
    DragonLib.make_parent_dir_if_not_exits(target_path=filepath)

    # state 정보 담기
    state = {
        ConstVar.KEY_STATE_MODEL_G: modelG.state_dict(),
        ConstVar.KEY_STATE_MODEL_D: modelD.state_dict(),
        ConstVar.KEY_STATE_OPTIMIZER_G: optimizerG.state_dict(),
        ConstVar.KEY_STATE_OPTIMIZER_D: optimizerD.state_dict(),
        ConstVar.KEY_STATE_EPOCH: epoch
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
    mean = torch.tensor(ConstVar.NORMALIZE_MEAN)
    std = torch.tensor(ConstVar.NORMALIZE_STD)
    plt_pics_list = [(
        (a.cpu().reshape(-1, 256, 256) * std[:, None, None] + mean[:, None, None]).permute(1, 2, 0),
        (b.cpu().reshape(-1, 256, 256) * std[:, None, None] + mean[:, None, None]).permute(1, 2, 0),
        (fake_b.cpu().detach().reshape(-1, 256, 256) * std[:, None, None] + mean[:, None, None]).permute(1, 2, 0)
    ) for a, b, fake_b in pics_list]

    # plt 에 그리기
    fig, axs = plt.subplots(nrows=ConstVar.NUM_PICS_LIST, ncols=3, figsize=(10, 15))
    fig.suptitle(t=title, fontsize=18)
    for num, (a, b, fake_b) in enumerate(plt_pics_list):
        axs[num, 0].imshow(X=a, cmap='gray')
        axs[num, 0].axis('off')
        axs[num, 1].imshow(X=b, cmap='gray')
        axs[num, 1].axis('off')
        axs[num, 2].imshow(X=fake_b, cmap='gray')
        axs[num, 2].axis('off')

        if num == 0:
            axs[num, 0].set_title('A')
            axs[num, 1].set_title('B')
            axs[num, 2].set_title('Fake B')

    # 저장하고자 하는 경로의 상위 디렉터리가 존재하지 않는 경우 상위 경로 생성
    DragonLib.make_parent_dir_if_not_exits(target_path=filepath)

    # 그림 저장
    plt.savefig(filepath)
