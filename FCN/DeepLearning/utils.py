import torch
import numpy as np
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
    :param pics_list: 입력, ground truth, predicted segmentation map, mIoU 시각화 이미지 쌍 담은 리스트. (사실 mIoU 시각화 이미지는 추후에 만들어질거임)
    :param filepath: 저장될 그림 파일 경로
    :param title: 그림 제목
    :return: 그림 파일 생성됨
    """

    # plt 로 시각화 할 수 있는 형식으로 변환된 이미지 담을 리스트
    plt_pics_list = []

    # plt 로 시각화 할 수 있는 형식으로 변환
    for x, y, y_pred in pics_list:
        x = x.cpu().reshape(-1, 224, 224).permute(1, 2, 0)
        y = y.cpu().reshape(-1, 224, 224).permute(1, 2, 0)
        y_pred = torch.argmax(input=y_pred,
                              dim=1)
        y_pred = y_pred.cpu().detach().reshape(-1, 224, 224).permute(1, 2, 0)
        mIoU_vis = visualize_mIoU(prediction_map=y_pred,
                                  ground_truth=y)

        plt_pics_list.append((x, y, y_pred, mIoU_vis))

    # plt 에 그리기
    fig, axs = plt.subplots(nrows=ConstVar.NUM_PICS_LIST, ncols=4, figsize=(10, 15))
    fig.suptitle(t=title, fontsize=18)
    for num, (x, y, y_pred, mIoU_vis) in enumerate(plt_pics_list):
        axs[num, 0].imshow(X=x, cmap='gray')
        axs[num, 0].axis('off')
        axs[num, 1].imshow(X=y, cmap='gray')
        axs[num, 1].axis('off')
        axs[num, 2].imshow(X=y_pred, cmap='gray')
        axs[num, 2].axis('off')
        axs[num, 3].imshow(X=mIoU_vis)
        axs[num, 3].axis('off')

        if num == 0:
            axs[num, 0].set_title('Input')
            axs[num, 1].set_title('Ground Truth')
            axs[num, 2].set_title('Prediction Map')
            axs[num, 3].set_title('mIoU')

    # 저장하고자 하는 경로의 상위 디렉터리가 존재하지 않는 경우 상위 경로 생성
    DragonLib.make_parent_dir_if_not_exits(target_path=filepath)

    # 그림 저장
    plt.savefig(filepath)


def visualize_mIoU(prediction_map, ground_truth):
    """
    * mIoU 시각화 진행
    :param prediction_map: 예측한 결과. (224, 224, batch * 1)
    :param ground_truth: 실제 mask. (224, 224, batch * 1)
    :return: mIoU 시각화 된 이미지. (224, 224, 3)
    """

    # 색상 표
    color_map = {
        # 겹치지 않는 구간
        "RED": (255, 0, 0),
        # 겹치는 구간
        "GREEN": (0, 255, 0)
    }

    # 3 채널 RGB 이미지로 시각화하기 위한 ndarray
    # (224, 224, 1) -> (224, 224, 3)
    mIoU_vis = np.zeros(shape=[*ground_truth.shape[:2], 3], dtype=np.uint8)

    # (224, 224, 1) -> (224, 224)
    prediction_map = prediction_map.numpy().squeeze()
    # (224, 224, 1) -> (224, 224)
    ground_truth = ground_truth.numpy().squeeze()

    # 전체 클래스 종류 조사
    total_classes = np.unique(ground_truth)

    # 클래스 별로 IoU 시각화
    for class_idx in total_classes:

        # 배경에 해당하는 클래스 (0) 은 건너뛰기
        if class_idx == 0:
            continue
        else:
            # prediction_map 에서 현재 클래스에 해당하는 픽셀 찾기
            prediction_map_template = np.where(prediction_map == class_idx, True, False)
            # ground_truth 에서 현재 클래스에 해당하는 픽셀 찾기
            ground_truth_template = np.where(ground_truth == class_idx, True, False)

            # predicted segmentation map 과 ground truth 의 교집합
            intersection = np.logical_and(prediction_map_template, ground_truth_template)
            # predicted segmentation map 과 ground truth 의 합집합
            union = np.logical_or(prediction_map_template, ground_truth_template)

            # 시각화 진행. 전체 구간 먼저 빨간 색으로 칠한 뒤 겹치는 구간은 초록색으로 덮어쓰기
            mIoU_vis[union] = color_map['RED']
            mIoU_vis[intersection] = color_map['GREEN']

    return mIoU_vis
