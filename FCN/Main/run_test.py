import os, sys, argparse, time


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
    parser = argparse.ArgumentParser(prog="Deep Learning Study Project Test",
                                     description="* Run this to test the model.")

    # parser 인자 목록 생성
    # 입력 이미지 파일 디렉터리 위치
    parser.add_argument("--input_dir",
                        type=str,
                        help='set input image file directory',
                        default=ConstVar.INPUT_DIR,
                        dest='input_dir')

    # 불러올 체크포인트 파일 경로
    parser.add_argument("--checkpoint_file",
                        type=str,
                        help='set checkpoint file to load if exists',
                        default='C:/dragonhyeon/python_directory/movements/40fcn/DATA/checkpoint/epoch00020.ckpt',
                        dest='checkpoint_file')

    # 결과물 파일 저장할 디렉터리 위치
    parser.add_argument("--output_dir",
                        type=str,
                        help='set the directory where output files will be saved',
                        default=ConstVar.OUTPUT_DIR,
                        dest='output_dir')

    # segmentation 된 이미지 파일 저장될 폴더명
    parser.add_argument("--segmentation_folder_name",
                        type=str,
                        help='set segmentation folder name which segmentation images going to be saved',
                        default=time.time(),
                        dest="segmentation_folder_name")

    # parsing 한거 가져오기
    args = parser.parse_args()

    return args


def run_program(args):
    """
    * 테스트 실행
    :param args: 프로그램 실행 인자
    :return: None
    """

    import torch

    from Common import ConstVar
    from DeepLearning.test import Tester
    from DeepLearning.model import FCNs, VGGNet
    from DeepLearning import utils

    # GPU / CPU 설정
    device = ConstVar.DEVICE_CUDA if torch.cuda.is_available() else ConstVar.DEVICE_CPU

    # 체크포인트 파일 불러오기
    state = utils.load_checkpoint(filepath=args.checkpoint_file)

    # 모델 선언 및 가중치 불러오기
    model = FCNs(pretrained_net=VGGNet(pretrained=True),
                 num_classes=ConstVar.NUM_CLASSES)
    model.load_state_dict(state[ConstVar.KEY_STATE_MODEL])
    # 모델을 해당 디바이스로 이동
    model.to(device)

    # 모델 테스트 객체 선언
    tester = Tester(model=model,
                    device=device)

    # 모델 테스트
    tester.running(input_dir=args.input_dir,
                   output_dir=args.output_dir,
                   segmentation_folder_name=args.segmentation_folder_name)


def main():

    # 경로 잡기
    set_path()

    # 인자 받기
    args = arguments()

    # 프로그램 실행
    run_program(args=args)


if __name__ == '__main__':
    main()
