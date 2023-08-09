import os

# 파일 경로
PROJECT_ROOT_DIRECTORY = os.path.dirname(os.getcwd())
DATA_DIR_TRAIN = '{0}/RES/facades/train/'.format(PROJECT_ROOT_DIRECTORY)
DATA_DIR_TEST = '{0}/RES/facades/test/'.format(PROJECT_ROOT_DIRECTORY)
OUTPUT_DIR = '{0}/DATA/'.format(PROJECT_ROOT_DIRECTORY)
OUTPUT_DIR_SUFFIX_CHECKPOINT = 'checkpoint'
OUTPUT_DIR_SUFFIX_PICS = 'pics'
OUTPUT_DIR_SUFFIX_GENERATED_IMG = 'generated_img/{0}'
OUTPUT_DIR_SUFFIX_ORIGINAL = 'original'
OUTPUT_DIR_SUFFIX_TRANSFERRED = 'transferred'
INPUT_DIR = '{0}/RES/sample/'.format(PROJECT_ROOT_DIRECTORY)
CHECKPOINT_FILE_NAME = 'epoch{:05d}.ckpt'
PICS_FILE_NAME = 'epoch{:05d}.png'
CHECKPOINT_BEST_FILE_NAME = 'best_model.ckpt'
GENERATED_IMG_FILE_NAME = '{0}.png'

# 학습 / 테스트 모드
MODE_TRAIN = 'train'
MODE_TEST = 'test'

# 데이터 변환 방향
A2B = 'a2b'
B2A = 'b2a'

# 디바이스 종류
DEVICE_CUDA = 'cuda'
DEVICE_CPU = 'cpu'

# 판별자 참, 거짓 정보
REAL_LABEL = 1
FAKE_LABEL = 0

# 하이퍼 파라미터
LEARNING_RATE = 0.0002
BATCH_SIZE = 4
NUM_EPOCH = 20
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

# 옵션 값
SHUFFLE = True
TRACKING_FREQUENCY = 1
NUM_PICS_LIST = 10

# 그 외 기본 설정 값
RESIZE_SIZE = 256
LAMBDA = 100

# state 저장시 딕셔너리 키 값
KEY_STATE_MODEL_G = 'modelG'
KEY_STATE_MODEL_D = 'modelD'
KEY_STATE_OPTIMIZER_G = 'optimizerG'
KEY_STATE_OPTIMIZER_D = 'optimizerD'
KEY_STATE_EPOCH = 'epoch'

# score 저장시 딕셔너리 키 값
KEY_SCORE_G = 'scoreG'
KEY_SCORE_D = 'scoreD'

# 초기 값
INITIAL_START_EPOCH_NUM = 1
INITIAL_BEST_BCE_LOSS = 100000
