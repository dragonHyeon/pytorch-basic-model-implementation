import os, sys, argparse


def set_path():
    """
    * 경로 잡기
    """

    # 경로명 설정
    AppPath = os.path.dirname(os.path.abspath(os.getcwd()))
    SRC_DIR_NAME_Common = 'Common'
    SRC_DIR_NAME_DATA = 'DATA'
    SRC_DIR_NAME_DeepLearning = 'DeepLearning'
    SRC_DIR_NAME_Lib = 'Lib'
    SRC_DIR_NAME_LOG = 'LOG'
    SRC_DIR_NAME_Main = 'Main'
    SRC_DIR_NAME_RES = 'RES'

    Common = os.path.join(AppPath, SRC_DIR_NAME_Common)
    DATA = os.path.join(AppPath, SRC_DIR_NAME_DATA)
    DeepLearning = os.path.join(AppPath, SRC_DIR_NAME_DeepLearning)
    Lib = os.path.join(AppPath, SRC_DIR_NAME_Lib)
    LOG = os.path.join(AppPath, SRC_DIR_NAME_LOG)
    Main = os.path.join(AppPath, SRC_DIR_NAME_Main)
    RES = os.path.join(AppPath, SRC_DIR_NAME_RES)

    # 경로 추가
    AppPathList = [AppPath, Common, DATA, DeepLearning, Lib, LOG, Main, RES]
    for p in AppPathList:
        sys.path.append(p)


def arguments():
    """
    * parser 이용하여 프로그램 실행 인자 받기
    :return: args
    """

    from Common import ConstVar

    # parser 생성
    parser = argparse.ArgumentParser(prog="Deep Learning Study Project Train",
                                     description="* Run this to train the model.")

    # parser 인자 목록 생성
    # 학습 데이터 디렉터리 설정
    parser.add_argument("--train_data_dir",
                        type=str,
                        help='set training data directory',
                        default=ConstVar.DATA_DIR_TRAIN,
                        dest="train_data_dir")

    # 결과물 파일 저장할 디렉터리 위치
    parser.add_argument("--output_dir",
                        type=str,
                        help='set the directory where output files will be saved',
                        default=ConstVar.OUTPUT_DIR,
                        dest='output_dir')

    # 체크포인트 파일 저장 및 학습 진행 기록 빈도수
    parser.add_argument("--tracking_frequency",
                        type=int,
                        help='set model training tracking frequency',
                        default=ConstVar.TRACKING_FREQUENCY,
                        dest='tracking_frequency')

    # 불러올 체크포인트 파일 경로
    parser.add_argument("--checkpoint_file",
                        type=str,
                        help='set checkpoint file to resume training if exists',
                        default=None,
                        dest='checkpoint_file')

    # learning rate 설정
    parser.add_argument("--learning_rate",
                        type=float,
                        help='set learning rate',
                        default=ConstVar.LEARNING_RATE,
                        dest='learning_rate')

    # 학습 반복 횟수
    parser.add_argument("--num_epoch",
                        type=int,
                        help='set number of epochs to train',
                        default=ConstVar.NUM_EPOCH,
                        dest='num_epoch')

    # parsing 한거 가져오기
    args = parser.parse_args()

    return args


def run_program(args):
    """
    * 학습 실행
    :param args: 프로그램 실행 인자
    :return: None
    """

    import torch
    from torch.utils.data import DataLoader

    from Common import ConstVar
    from Lib import UtilLib
    from DeepLearning import utils
    from DeepLearning.train import Trainer
    from DeepLearning.model import VGGNet
    from DeepLearning.loss import content_loss, StyleLoss
    from DeepLearning.metric import accuracy

    # GPU / CPU 설정
    device = ConstVar.DEVICE_CUDA if torch.cuda.is_available() else ConstVar.DEVICE_CPU

    # 모델이 아닌 이미지 선언 및 이미지를 해당 디바이스로 이동
    original_image = utils.read_img(filepath=UtilLib.getNewPath(path=args.train_data_dir, add=ConstVar.CONTENT_IMAGE_FILE_NAME)).to(device).requires_grad_(True)

    # 데이터 선언
    content_image = utils.read_img(filepath=UtilLib.getNewPath(path=args.train_data_dir, add=ConstVar.CONTENT_IMAGE_FILE_NAME))
    style_image = utils.read_img(filepath=UtilLib.getNewPath(path=args.train_data_dir, add=ConstVar.STYLE_IMAGE_FILE_NAME))

    # 모델 선언
    model = VGGNet()
    # 모델을 해당 디바이스로 이동
    model.to(device)
    # 모델을 테스트 모드로 전환
    model.eval()
    # 모델 파라미터 freeze
    for param in model.parameters():
        param.requires_grad = False

    # optimizer 선언
    optimizer = torch.optim.Adam(params=[original_image],
                                 lr=args.learning_rate)

    # loss 선언
    style_loss = StyleLoss()

    # 모델 학습 객체 선언
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      content_loss=content_loss,
                      style_loss=style_loss,
                      metric_fn=None,
                      original_image=original_image,
                      content_image=content_image,
                      style_image=style_image,
                      device=device)

    # 모델 학습
    trainer.running(num_epoch=args.num_epoch,
                    output_dir=args.output_dir,
                    train_data_dir=args.train_data_dir,
                    tracking_frequency=args.tracking_frequency,
                    checkpoint_file=args.checkpoint_file,
                    learning_rate=args.learning_rate)


def main():

    # 경로 잡기
    set_path()

    # 인자 받기
    args = arguments()

    # 프로그램 실행
    run_program(args=args)


if __name__ == '__main__':
    main()
