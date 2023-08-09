import numpy as np


def mIoU(y_pred, y):
    """
    * 해당 배치의 mIoU 구하기
    :param y_pred: shape: (N, num_classes, 224, 224)
    :param y: shape: (N, 224, 224)
    :return: 해당하는 배치의 mIoU
    """

    # y (ground truth) 와 비교 할 수 있게 같은 y_pred 를 y 와 같은 형태로 만들기
    # ex.
    # (N, 2, 2, 2) -> (N, 2, 2)
    # [[[2.5, 0.3], [0.2, 5.1]], [[4.1, 5.2], [9.3, 13.7]]] -> [[1, 1], [1, 1]]
    y_pred_map = np.argmax(a=y_pred.tolist(),
                           axis=1)
    y = y.cpu().numpy()

    # mIoU 계산
    # 클래스별 IoU 담을 dict
    IoU_dict = dict()
    # 전체 클래스 종류 조사
    total_classes = np.unique(y)

    # 클래스 별로 IoU 계산
    for class_idx in total_classes:

        # 배경에 해당하는 클래스 (0) 은 건너뛰기
        if class_idx == 0:
            continue
        else:
            # y_pred 에서 현재 클래스에 해당하는 픽셀 찾기
            y_pred_map_template = np.where(y_pred_map == class_idx, True, False)
            # y 에서 현재 클래스에 해당하는 픽셀 찾기
            y_template = np.where(y == class_idx, True, False)

            # predicted segmentation map 과 ground truth 의 교집합
            intersection = np.logical_and(y_pred_map_template, y_template)
            # predicted segmentation map 과 ground truth 의 합집합
            union = np.logical_or(y_pred_map_template, y_template)

            # 해당 클래스가 없는 경우에는 계산 하지 않고 넘어가기 (사실 np.unique 로 존재하는 클래스에 대해서만 조사를 하였기 때문에 필요 없는 처리이지만 안정성을 위해 작성)
            if np.sum(union) == 0:
                continue
            else:
                # 해당 클래스에 대한 IoU 계산
                IoU_dict[class_idx] = np.sum(intersection) / np.sum(union)

    # mIoU 계산
    # sum(클래스 별 IoU) / 클래스 개수
    batch_mIoU = np.average(list(IoU_dict.values()))

    return batch_mIoU
