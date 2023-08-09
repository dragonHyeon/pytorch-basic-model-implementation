import shutil

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
    def __init__(self, model, optimizer, content_loss, style_loss, metric_fn, original_image, content_image, style_image, device):
        """
        * 학습 관련 클래스
        :param model: 학습 시킬 모델
        :param optimizer: 학습 optimizer
        :param content_loss: 손실 함수 (content loss)
        :param style_loss: 손실 함수 (style loss)
        :param metric_fn: 성능 평가 지표
        :param original_image: 원본 이미지
        :param content_image: 유지할 이미지
        :param style_image: 스타일 이미지
        :param device: GPU / CPU
        """

        # 학습 시킬 모델
        self.model = model
        # 학습 optimizer
        self.optimizer = optimizer
        # 손실 함수
        self.content_loss = content_loss
        self.style_loss = style_loss
        # 성능 평가 지표
        self.metric_fn = metric_fn
        # 이미지 데이터
        self.original_image = original_image
        self.content_image = content_image
        self.style_image = style_image
        # GPU / CPU
        self.device = device

    def running(self, num_epoch, output_dir, train_data_dir, tracking_frequency, **checkpoint_dict):
        """
        * 학습 셋팅 및 진행
        :param num_epoch: 학습 반복 횟수
        :param output_dir: 결과물 파일 저장할 디렉터리 위치
        :param train_data_dir: 학습한 해당 이미지 디렉터리 (체크포인트 및 학습 결과물 저장시 사용)
        :param tracking_frequency: 체크포인트 파일 저장 및 학습 진행 기록 빈도수
        :param checkpoint_dict: 불러올 체크포인트 파일 및 optimizer 다시 선언하기 위한 learning rate 담은 dict. ({'checkpoint_file': , 'learning_rate': })
        :return: 학습 완료 및 체크포인트 파일 생성됨
        """

        # 체크포인트 및 학습 결과물 저장 관련 경로 및 디렉터리
        directory_name = DragonLib.get_bottom_folder(path=train_data_dir)
        checkpoint_dir = UtilLib.getNewPath(path=output_dir, add=ConstVar.OUTPUT_DIR_SUFFIX_CHECKPOINT.format(directory_name))
        pics_dir = UtilLib.getNewPath(path=output_dir, add=ConstVar.OUTPUT_DIR_SUFFIX_PICS.format(directory_name))
        # 학습 이미지 경로
        content_img_filepath = UtilLib.getNewPath(path=train_data_dir, add=ConstVar.CONTENT_IMAGE_FILE_NAME)
        style_img_filepath = UtilLib.getNewPath(path=train_data_dir, add=ConstVar.STYLE_IMAGE_FILE_NAME)
        # 체크포인트 학습 이미지 저장 경로
        checkpoint_content_img_filepath = UtilLib.getNewPath(path=checkpoint_dir, add=ConstVar.CONTENT_IMAGE_FILE_NAME)
        checkpoint_style_img_filepath = UtilLib.getNewPath(path=checkpoint_dir, add=ConstVar.STYLE_IMAGE_FILE_NAME)

        # epoch 초기화
        start_epoch_num = ConstVar.INITIAL_START_EPOCH_NUM

        # 불러올 체크포인트 파일 있을 경우 불러오기
        if checkpoint_dict[ConstVar.KEY_CHECKPOINT_FILE]:
            state = utils.load_checkpoint(filepath=checkpoint_dict[ConstVar.KEY_CHECKPOINT_FILE])
            self.original_image = state[ConstVar.KEY_STATE_ORIGINAL_IMAGE].to(self.device)
            # optimizer 다시 선언해주기
            self.optimizer = torch.optim.Adagrad(params=[self.original_image], lr=checkpoint_dict[ConstVar.KEY_LEARNING_RATE])
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
                score, transferred_image = self._eval()

                # 체크포인트 저장
                checkpoint_filepath = UtilLib.getNewPath(path=checkpoint_dir, add=ConstVar.CHECKPOINT_FILE_NAME.format(current_epoch_num))
                utils.save_checkpoint(filepath=checkpoint_filepath,
                                      original_image=self.original_image,
                                      optimizer=self.optimizer,
                                      epoch=current_epoch_num,
                                      score=score,
                                      is_best=self._check_is_best(score=score,
                                                                  best_checkpoint_dir=checkpoint_dir))

                # 그래프 시각화 진행
                self._draw_graph(score=score,
                                 current_epoch_num=current_epoch_num,
                                 title='Loss Progress')

                # 결과물 시각화 진행
                pics_filepath = UtilLib.getNewPath(path=pics_dir, add=ConstVar.PICS_FILE_NAME.format(current_epoch_num))
                self._draw_pic(transferred_image=transferred_image,
                               content_image=self.content_image.clone(),
                               style_image=self.style_image.clone(),
                               title='Epoch {0}'.format(current_epoch_num),
                               filepath=pics_filepath)

        # 체크포인트 디렉터리에 학습에 사용된 이미지도 함께 저장
        shutil.copyfile(src=content_img_filepath, dst=checkpoint_content_img_filepath)
        shutil.copyfile(src=style_img_filepath, dst=checkpoint_style_img_filepath)

    def _train(self):
        """
        * 학습 진행
        :return: 1 epoch 만큼 학습 진행
        """

        # 각 텐서를 해당 디바이스로 이동
        self.content_image = self.content_image.to(self.device)
        self.style_image = self.style_image.to(self.device)

        # feature 추출 진행
        original_features = self.model(self.original_image)
        content_features = self.model(self.content_image)
        style_features = self.model(self.style_image)

        # Content loss
        loss_content = self.content_loss(original_features=original_features, content_features=content_features)

        # Style loss
        loss_style = self.style_loss(original_features=original_features, style_features=style_features)

        # Total loss
        loss = ConstVar.LAMBDA_CONTENT * loss_content + ConstVar.LAMBDA_STYLE * loss_style

        # 역전파
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _eval(self):
        """
        * 테스트 진행
        :return: 현재 epoch 성능 평가 점수, 학습 결과물
        """

        # 테스트를 위해 데이터 복제
        test_image = self.original_image.clone()

        # 각 텐서를 해당 디바이스로 이동
        test_image = test_image.to(self.device)
        self.content_image = self.content_image.to(self.device)
        self.style_image = self.style_image.to(self.device)

        # feature 추출 진행
        test_features = self.model(test_image)
        content_features = self.model(self.content_image)
        style_features = self.model(self.style_image)

        # Content loss
        loss_content = self.content_loss(original_features=test_features, content_features=content_features)

        # Style loss
        loss_style = self.style_loss(original_features=test_features, style_features=style_features)

        # Total loss
        loss = ConstVar.LAMBDA_CONTENT * loss_content + ConstVar.LAMBDA_STYLE * loss_style

        return loss, test_image

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
            # 없다면 임의의 큰 숫자 (100000) 로 초기화
            else:
                self.best_score = ConstVar.INITIAL_BEST_LOSS

        # best 성능 갱신
        if score < self.best_score:
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
    def _draw_pic(transferred_image, content_image, style_image, title, filepath):
        """
        * 학습 결과물 이미지로 저장
        :param transferred_image: 변환된 이미지
        :param content_image: 유지할 이미지
        :param style_image: 스타일 이미지
        :param title: 그림 제목
        :param filepath: 저장될 그림 파일 경로
        :return: 그림 파일 생성됨
        """

        # 시각화를 위해 standardization 한 거 원래대로 되돌리기
        # plt 에 맞게 (N (1), C, H, W) -> (H, W, C) 변환
        mean = torch.tensor(ConstVar.NORMALIZE_MEAN)
        std = torch.tensor(ConstVar.NORMALIZE_STD)
        transferred_image_plt = (transferred_image[0].cpu().detach() * std[:, None, None] + mean[:, None, None]).permute(1, 2, 0)
        content_image_plt = (content_image[0].cpu().detach() * std[:, None, None] + mean[:, None, None]).permute(1, 2, 0)
        style_image_plt = (style_image[0].cpu().detach() * std[:, None, None] + mean[:, None, None]).permute(1, 2, 0)

        # 시각화 진행
        fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
        fig.suptitle(t=title, fontsize=18)
        axs[0].imshow(X=content_image_plt, cmap='gray')
        axs[0].set_title(label='Content image', fontdict={'fontsize': 13})
        axs[0].axis('off')
        axs[1].imshow(X=transferred_image_plt, cmap='gray')
        axs[1].set_title(label='Content image + Style image', fontdict={'fontsize': 13})
        axs[1].axis('off')
        axs[2].imshow(X=style_image_plt, cmap='gray')
        axs[2].set_title(label='Style image', fontdict={'fontsize': 13})
        axs[2].axis('off')

        # 저장하고자 하는 경로의 상위 디렉터리가 존재하지 않는 경우 상위 경로 생성
        DragonLib.make_parent_dir_if_not_exits(target_path=filepath)

        # 그림 저장
        plt.savefig(filepath)
