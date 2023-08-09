import visdom
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

from Lib import UtilLib, DragonLib
from Common import ConstVar
from DeepLearning import utils


class Trainer:
    def __init__(self, modelG, modelD, optimizerG, optimizerD, loss_fn, metric_fn, train_dataloader, test_dataloader, device):
        """
        * 학습 관련 클래스
        :param modelG: 학습 시킬 모델. 생성자
        :param modelD: 학습 시킬 모델. 판별자
        :param optimizerG: 생성자 학습 optimizer
        :param optimizerD: 판별자 학습 optimizer
        :param loss_fn: 손실 함수
        :param metric_fn: 성능 평가 지표
        :param train_dataloader: 학습용 데이터로더
        :param test_dataloader: 테스트용 데이터로더
        :param device: GPU / CPU
        """

        # 학습 시킬 모델
        self.modelG = modelG
        self.modelD = modelD
        # 학습 optimizer
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD
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

        # 각 모델 가중치 초기화
        self.modelG.apply(weights_init)
        self.modelD.apply(weights_init)

        # 불러올 체크포인트 파일 있을 경우 불러오기
        if checkpoint_file:
            state = utils.load_checkpoint(filepath=checkpoint_file)
            self.modelG.load_state_dict(state[ConstVar.KEY_STATE_MODEL_G])
            self.modelD.load_state_dict(state[ConstVar.KEY_STATE_MODEL_D])
            self.optimizerG.load_state_dict(state[ConstVar.KEY_STATE_OPTIMIZER_G])
            self.optimizerD.load_state_dict(state[ConstVar.KEY_STATE_OPTIMIZER_D])
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
                score, pics_list = self._eval()

                # 체크포인트 저장
                checkpoint_dir = UtilLib.getNewPath(path=output_dir, add=ConstVar.OUTPUT_DIR_SUFFIX_CHECKPOINT)
                checkpoint_filepath = UtilLib.getNewPath(path=checkpoint_dir, add=ConstVar.CHECKPOINT_FILE_NAME.format(current_epoch_num))
                utils.save_checkpoint(filepath=checkpoint_filepath,
                                      modelG=self.modelG,
                                      modelD=self.modelD,
                                      optimizerG=self.optimizerG,
                                      optimizerD=self.optimizerD,
                                      epoch=current_epoch_num,
                                      is_best=False)

                # 그래프 시각화 진행
                self._draw_graph(score=score,
                                 current_epoch_num=current_epoch_num,
                                 title=nn.BCELoss.__name__)

                # 결과물 시각화 진행
                pics_dir = UtilLib.getNewPath(path=output_dir, add=ConstVar.OUTPUT_DIR_SUFFIX_PICS)
                pics_filepath = UtilLib.getNewPath(path=pics_dir, add=ConstVar.PICS_FILE_NAME.format(current_epoch_num))
                self._draw_pic(pics_list=pics_list,
                               title='Epoch {0}'.format(current_epoch_num),
                               filepath=pics_filepath)

    def _train(self):
        """
        * 학습 진행
        :return: 1 epoch 만큼 학습 진행
        """

        # 각 모델을 학습 모드로 전환
        self.modelG.train()
        self.modelD.train()

        # loss 선언
        bce = nn.BCELoss()
        l1 = nn.L1Loss()

        # a shape: (N, 3, 256, 256)
        # b shape: (N, 3, 256, 256)
        for a, b in tqdm(self.train_dataloader, desc='train dataloader', leave=False):

            # 현재 배치 사이즈
            batch_size = a.shape[0]
            # 패치 한 개 사이즈
            patch_size = (1, int(a.shape[2] / 16), int(a.shape[3] / 16))

            # real image label
            real_label = torch.ones(size=(batch_size, *patch_size), device=self.device)
            # fake image label
            fake_label = torch.zeros(size=(batch_size, *patch_size), device=self.device)

            # 각 텐서를 해당 디바이스로 이동
            a = a.to(self.device)
            b = b.to(self.device)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # GAN loss
            lossD_GAN_real = bce(self.modelD(b, a), real_label)
            fake_b = self.modelG(a)
            lossD_GAN_fake = bce(self.modelD(fake_b.detach(), a), fake_label)
            lossD_GAN = lossD_GAN_real + lossD_GAN_fake

            # Total loss
            lossD = lossD_GAN

            # 역전파
            self.modelD.zero_grad()
            lossD.backward()
            self.optimizerD.step()

            # -----------------
            #  Train Generator
            # -----------------

            # GAN loss
            lossG_GAN = bce(self.modelD(fake_b, a), real_label)

            # L1 loss
            lossG_L1 = l1(fake_b, b)

            # Total loss
            lossG = lossG_GAN + ConstVar.LAMBDA * lossG_L1

            # 역전파
            self.modelG.zero_grad()
            lossG.backward()
            self.optimizerG.step()

    def _eval(self):
        """
        * 테스트 진행
        :return: 현재 epoch 성능 평가 점수, 학습 결과물
        """

        # 각 모델을 테스트 모드로 전환
        self.modelG.eval()
        self.modelD.eval()

        # loss 선언
        bce = nn.BCELoss()
        l1 = nn.L1Loss()

        # 배치 마다의 G / D score 담을 리스트
        batch_score_listG = list()
        batch_score_listD = list()

        # 생성된 이미지 담을 리스트
        pics_list = list()

        # a shape: (N (1), 3, 256, 256)
        # b shape: (N (1), 3, 256, 256)
        for a, b in tqdm(self.test_dataloader, desc='test dataloader', leave=False):

            # 현재 배치 사이즈
            batch_size = a.shape[0]
            # 패치 한 개 사이즈
            patch_size = (1, int(a.shape[2] / 16), int(a.shape[3] / 16))

            # real image label
            real_label = torch.ones(size=(batch_size, *patch_size), device=self.device)
            # fake image label
            fake_label = torch.zeros(size=(batch_size, *patch_size), device=self.device)

            # 각 텐서를 해당 디바이스로 이동
            a = a.to(self.device)
            b = b.to(self.device)

            # --------------------
            #  Test Discriminator
            # --------------------

            # GAN loss
            scoreD_GAN_real = bce(self.modelD(b, a), real_label)
            fake_b = self.modelG(a)
            scoreD_GAN_fake = bce(self.modelD(fake_b, a), fake_label)
            scoreD_GAN = scoreD_GAN_real + scoreD_GAN_fake

            # Total loss
            scoreD = scoreD_GAN

            # 배치 마다의 판별자 D score 계산
            batch_score_listD.append(scoreD.cpu().detach())

            # ----------------
            #  Test Generator
            # ----------------

            # GAN loss
            scoreG_GAN = bce(self.modelD(fake_b, a), real_label)

            # L1 loss
            scoreG_L1 = l1(fake_b, b)

            # Total loss
            scoreG = scoreG_GAN + ConstVar.LAMBDA * scoreG_L1

            # 배치 마다의 생성자 G score 계산
            batch_score_listG.append(scoreG.cpu().detach())

            # a, b, fake_b 이미지 쌍 담기 (설정한 개수 만큼)
            if len(pics_list) < ConstVar.NUM_PICS_LIST:
                pics_list.append((a, b, fake_b))

        # score 기록
        score = {
            ConstVar.KEY_SCORE_G: np.mean(batch_score_listG),
            ConstVar.KEY_SCORE_D: np.mean(batch_score_listD)
        }

        return score, pics_list

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
                self.best_score = state[ConstVar.KEY_SCORE]
            # 없다면 임의의 큰 숫자 (100000) 로 초기화
            else:
                self.best_score = ConstVar.INITIAL_BEST_BCE_LOSS

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
            self.vis.line(Y=torch.cat((torch.Tensor([score[ConstVar.KEY_SCORE_G]]).view(-1, 1), torch.Tensor([score[ConstVar.KEY_SCORE_D]]).view(-1, 1)), dim=1),
                          X=torch.cat((torch.Tensor([current_epoch_num]).view(-1, 1), torch.Tensor([current_epoch_num]).view(-1, 1)), dim=1),
                          win=self.plt,
                          update='append',
                          opts=dict(title=title,
                                    legend=['G loss', 'D loss'],
                                    showlegend=True))
        except AttributeError:
            self.plt = self.vis.line(Y=torch.cat((torch.Tensor([score[ConstVar.KEY_SCORE_G]]).view(-1, 1), torch.Tensor([score[ConstVar.KEY_SCORE_D]]).view(-1, 1)), dim=1),
                                     X=torch.cat((torch.Tensor([current_epoch_num]).view(-1, 1), torch.Tensor([current_epoch_num]).view(-1, 1)), dim=1),
                                     opts=dict(title=title,
                                               legend=['G loss', 'D loss'],
                                               showlegend=True))

    def _draw_pic(self, pics_list, title, filepath):
        """
        * 학습 결과물 이미지로 저장
        :param pics_list: 원본, 재구성 이미지 쌍 담은 리스트
        :param title: 그림 제목
        :param filepath: 저장될 그림 파일 경로
        :return: 그림 파일 생성됨
        """

        # 시각화를 위해 standardization 한 거 원래대로 되돌리기
        # plt 에 맞게 (N, C, H, W) -> (N, H, W, C) 변환
        mean = torch.tensor(ConstVar.NORMALIZE_MEAN)
        std = torch.tensor(ConstVar.NORMALIZE_STD)
        plt_pics_list = [(
            (a.cpu().reshape(-1, ConstVar.RESIZE_SIZE, ConstVar.RESIZE_SIZE) * std[:, None, None] + mean[:, None, None]).permute(1, 2, 0),
            (b.cpu().reshape(-1, ConstVar.RESIZE_SIZE, ConstVar.RESIZE_SIZE) * std[:, None, None] + mean[:, None, None]).permute(1, 2, 0),
            (fake_b.cpu().detach().reshape(-1, ConstVar.RESIZE_SIZE, ConstVar.RESIZE_SIZE) * std[:, None, None] + mean[:, None, None]).permute(1, 2, 0)
        ) for a, b, fake_b in pics_list]

        # 시각화 진행
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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
