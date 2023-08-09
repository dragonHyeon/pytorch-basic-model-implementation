import visdom
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from Lib import UtilLib, DragonLib
from Common import ConstVar
from DeepLearning import utils


class Trainer:
    def __init__(self, model, optimizer, loss_fn, metric_fn, train_dataloader, test_dataloader, device):
        """
        * 학습 관련 클래스
        :param model: 학습 시킬 모델
        :param optimizer: 학습 optimizer
        :param loss_fn: 손실 함수
        :param metric_fn: 성능 평가 지표
        :param train_dataloader: 학습용 데이터로더
        :param test_dataloader: 테스트용 데이터로더
        :param device: GPU / CPU
        """

        # 학습 시킬 모델
        self.model = model
        # 학습 optimizer
        self.optimizer = optimizer
        # 손실 함수
        self.loss_fn = loss_fn
        # 성능 평가 지표
        self.metric_fn = metric_fn
        # 학습용 데이터로더
        self.train_dataloader = train_dataloader
        # 테스트용 데이터로더
        self.test_dataloader = test_dataloader
        # GPU / CPU
        self.device = device

    def running(self, num_epoch, output_dir, tracking_frequency, checkpoint_file=None):
        """
        * 학습 셋팅 및 진행
        :param num_epoch: 학습 반복 횟수
        :param output_dir: 결과물 파일 저장할 디렉터리 위치
        :param tracking_frequency: 체크포인트 파일 저장 및 학습 진행 기록 빈도수
        :param checkpoint_file: 불러올 체크포인트 파일
        :return: 학습 완료 및 체크포인트 파일 생성됨
        """

        # epoch 초기화
        start_epoch_num = ConstVar.INITIAL_START_EPOCH_NUM

        # 불러올 체크포인트 파일 있을 경우 불러오기
        if checkpoint_file:
            state = utils.load_checkpoint(filepath=checkpoint_file)
            self.model.load_state_dict(state[ConstVar.KEY_STATE_MODEL])
            self.optimizer.load_state_dict(state[ConstVar.KEY_STATE_OPTIMIZER])
            start_epoch_num = state[ConstVar.KEY_STATE_EPOCH] + 1

        # num epoch 만큼 학습 반복
        for current_epoch_num in tqdm(range(start_epoch_num, num_epoch + 1),
                                      desc='training process',
                                      total=num_epoch,
                                      initial=start_epoch_num - 1):

            # 학습 진행
            self._train()

            # 학습 진행 기록 주기마다 학습 진행 저장 및 시각화
            if current_epoch_num % tracking_frequency == 0:

                # 테스트 진행
                score, result_list = self._eval()

                # 체크포인트 저장
                checkpoint_dir = UtilLib.getNewPath(path=output_dir, add=ConstVar.OUTPUT_DIR_SUFFIX_CHECKPOINT)
                checkpoint_filepath = UtilLib.getNewPath(path=checkpoint_dir, add=ConstVar.CHECKPOINT_FILE_NAME.format(current_epoch_num))
                utils.save_checkpoint(filepath=checkpoint_filepath,
                                      model=self.model,
                                      optimizer=self.optimizer,
                                      epoch=current_epoch_num,
                                      score=score,
                                      is_best=self._check_is_best(score=score,
                                                                  best_checkpoint_dir=checkpoint_dir))

                # 그래프 시각화 진행
                self._draw_graph(score=score,
                                 current_epoch_num=current_epoch_num,
                                 title=self.metric_fn.__name__)

                # 결과물 시각화 진행
                pics_dir = UtilLib.getNewPath(path=output_dir, add=ConstVar.OUTPUT_DIR_SUFFIX_PICS)
                pics_filepath = UtilLib.getNewPath(path=pics_dir, add=ConstVar.PICS_FILE_NAME.format(current_epoch_num))
                self._draw_pic(result_list=result_list,
                               title='Epoch {0}'.format(current_epoch_num),
                               filepath=pics_filepath)

    def _train(self):
        """
        * 학습 진행
        :return: 1 epoch 만큼 학습 진행
        """

        # 모델을 학습 모드로 전환
        self.model.train()

        # loss 선언
        cross_entropy = nn.CrossEntropyLoss()

        # x shape: (N, 3, 224, 224)
        # y shape: (N)
        for x, y in tqdm(self.train_dataloader, desc='train dataloader', leave=False):

            # 각 텐서를 해당 디바이스로 이동
            x = x.to(self.device)
            y = y.to(self.device)

            # 순전파
            y_pred = self.model(x)
            loss = cross_entropy(y_pred, y)

            # 역전파
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _eval(self):
        """
        * 테스트 진행
        :return: 현재 epoch 성능 평가 점수, 학습 결과물
        """

        # 모델을 테스트 모드로 전환
        self.model.eval()

        # 배치 마다의 정확도 담을 리스트
        batch_accuracy_list = list()

        # 생성된 결과물 담을 리스트
        result_list = list()

        # x shape: (N (1), 3, 224, 224)
        # y shape: (N (1))
        for x, y in tqdm(self.test_dataloader, desc='test dataloader', leave=False):

            # 각 텐서를 해당 디바이스로 이동
            x = x.to(self.device)
            y = y.to(self.device)

            # 순전파
            y_pred = self.model(x)

            # 배치 마다의 정확도 계산
            batch_accuracy_list.append(self.metric_fn(y_pred=y_pred, y=y))

            # x, y_pred, y 쌍 담기 (설정한 개수 만큼)
            if len(result_list) < ConstVar.NUM_RESULT_LIST:
                result_list.append((x, y_pred, y))

        # score 기록
        score = np.mean(batch_accuracy_list)

        return score, result_list

    def _check_is_best(self, score, best_checkpoint_dir):
        """
        * 현재 저장하려는 모델이 가장 좋은 성능의 모델인지 여부 확인
        :param score: 현재 모델의 성능 점수
        :param best_checkpoint_dir: 비교할 best 체크포인트 파일 디렉터리 위치
        :return: True / False
        """

        # best 성능 측정을 위해 초기화
        try:
            self.best_score
        except AttributeError:
            checkpoint_file = UtilLib.getNewPath(path=best_checkpoint_dir,
                                                 add=ConstVar.CHECKPOINT_BEST_FILE_NAME)
            # 기존에 측정한 best 체크포인트가 있으면 해당 score 로 초기화
            if UtilLib.isExist(checkpoint_file):
                state = utils.load_checkpoint(filepath=checkpoint_file)
                self.best_score = state[ConstVar.KEY_STATE_SCORE]
            # 없다면 0 으로 초기화
            else:
                self.best_score = ConstVar.INITIAL_BEST_ACCURACY_ZERO

        # best 성능 갱신
        if score > self.best_score:
            self.best_score = score
            return True
        else:
            return False

    def _draw_graph(self, score, current_epoch_num, title):
        """
        * 학습 진행 상태 실시간으로 시각화
        :param score: 성능 평가 점수
        :param current_epoch_num: 현재 에폭 수
        :param title: 그래프 제목
        :return: visdom 으로 시각화 진행
        """

        # 서버 켜기
        try:
            self.vis
        except AttributeError:
            self.vis = visdom.Visdom()
        # 실시간으로 학습 진행 상태 그리기
        try:
            self.vis.line(Y=torch.Tensor([score]),
                          X=torch.Tensor([current_epoch_num]),
                          win=self.plt,
                          update='append',
                          opts=dict(title=title))
        except AttributeError:
            self.plt = self.vis.line(Y=torch.Tensor([score]),
                                     X=torch.Tensor([current_epoch_num]),
                                     opts=dict(title=title))

    @staticmethod
    def _draw_pic(result_list, title, filepath):
        """
        * 학습 결과물 이미지로 저장
        :param result_list: x, y_pred, y 쌍 담은 리스트
        :param title: 그림 제목
        :param filepath: 저장될 그림 파일 경로
        :return: 그림 파일 생성됨
        """

        # 시각화를 위해 standardization 한 거 원래대로 되돌리기
        # plt 에 맞게 (N, C, H, W) -> (N, H, W, C) 변환
        # y_pred 는 대표값으로, y 는 데이터만 뽑기
        mean = torch.tensor(ConstVar.NORMALIZE_MEAN)
        std = torch.tensor(ConstVar.NORMALIZE_STD)
        plt_result_list = [(
            (x.cpu().reshape(-1, ConstVar.RESIZE_SIZE, ConstVar.RESIZE_SIZE) * std[:, None, None] + mean[:, None, None]).permute(1, 2, 0),
            torch.argmax(input=y_pred, dim=1).item(),
            y.item()
        ) for x, y_pred, y in result_list]

        # 시각화 진행
        fig, axs = plt.subplots(nrows=10, ncols=2, figsize=(3, 7))
        fig.suptitle(t=title, fontsize=18)
        for num, (x, y_pred, y) in enumerate(plt_result_list):

            # 이미지
            axs[num, 0].imshow(X=x, cmap='gray')
            axs[num, 0].axis('off')

            # classification 결과
            s = 'result: {0}\nanswer: {1}'.format(utils.idx_to_class(idx=y_pred), utils.idx_to_class(idx=y))
            axs[num, 1].text(x=0, y=0.5, s=s, color='green' if y_pred == y else 'red', fontsize=7, horizontalalignment='left', verticalalignment='center')
            axs[num, 1].axis('off')

        # 저장하고자 하는 경로의 상위 디렉터리가 존재하지 않는 경우 상위 경로 생성
        DragonLib.make_parent_dir_if_not_exits(target_path=filepath)

        # 그림 저장
        plt.savefig(filepath)
