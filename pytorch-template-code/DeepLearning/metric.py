import numpy as np


def accuracy(y_pred, y):
    """
    * 해당 배치의 예측 정확도 구하기
    :param y_pred: shape: (batch, label 개수)
    :param y: shape: (batch)
    :return: 해당하는 배치의 예측 정확도
    """

    # tensor 를 ndarray 로 변환
    # 배치 내 각각의 결과에 대해 가장 큰 값으로 output 설정 (ex. [0.1, 0.1, 0.8, 0] -> 2)
    y_pred = np.argmax(a=y_pred.tolist(),
                       axis=1)
    y = y.cpu().numpy()

    # 정확도 계산
    batch_accuracy = np.sum(y_pred == y) / len(y)

    return batch_accuracy
