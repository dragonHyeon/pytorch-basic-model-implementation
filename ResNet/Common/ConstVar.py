import os

# 파일 경로
PROJECT_ROOT_DIRECTORY = os.path.dirname(os.getcwd())
DATA_DIR_TRAIN = '{0}/RES/CIFAR-100 dataset/train/'.format(PROJECT_ROOT_DIRECTORY)
DATA_DIR_TEST = '{0}/RES/CIFAR-100 dataset/test/'.format(PROJECT_ROOT_DIRECTORY)
OUTPUT_DIR = '{0}/DATA/'.format(PROJECT_ROOT_DIRECTORY)
OUTPUT_DIR_SUFFIX_CHECKPOINT = 'checkpoint'
OUTPUT_DIR_SUFFIX_PICS = 'pics'
INPUT_PATH = '{0}/RES/sample/sample.jpg'.format(PROJECT_ROOT_DIRECTORY)
CHECKPOINT_FILE_NAME = 'epoch{:05d}.ckpt'
PICS_FILE_NAME = 'epoch{:05d}.png'
CHECKPOINT_BEST_FILE_NAME = 'best_model.ckpt'

# 학습 / 테스트 모드
MODE_TRAIN = 'train'
MODE_TEST = 'test'

# 디바이스 종류
DEVICE_CUDA = 'cuda'
DEVICE_CPU = 'cpu'

# 하이퍼 파라미터
LEARNING_RATE = 0.001
BATCH_SIZE = 8
NUM_EPOCH = 20
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

# 옵션 값
SHUFFLE = True
TRACKING_FREQUENCY = 1
NUM_PICS_LIST = 10
NUM_RESULT_LIST = 10

# 그 외 기본 설정 값
RESIZE_SIZE = 224

# state 저장시 딕셔너리 키 값
KEY_STATE_MODEL = 'model'
KEY_STATE_OPTIMIZER = 'optimizer'
KEY_STATE_EPOCH = 'epoch'
KEY_STATE_SCORE = 'score'

# 초기 값
INITIAL_START_EPOCH_NUM = 1
INITIAL_BEST_ACCURACY_ZERO = 0
