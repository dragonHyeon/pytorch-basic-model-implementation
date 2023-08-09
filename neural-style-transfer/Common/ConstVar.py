import os

# 파일 경로
PROJECT_ROOT_DIRECTORY = os.path.dirname(os.getcwd())
DATA_DIR_TRAIN = '{0}/RES/veteran2webtoon/'.format(PROJECT_ROOT_DIRECTORY)
OUTPUT_DIR = '{0}/DATA/'.format(PROJECT_ROOT_DIRECTORY)
OUTPUT_DIR_SUFFIX_CHECKPOINT = 'checkpoint/{0}'
OUTPUT_DIR_SUFFIX_PICS = 'pics/{0}'
OUTPUT_DIR_SUFFIX_GENERATED_IMG = 'generated_img/{0}'
CONTENT_IMAGE_FILE_NAME = 'content_image.png'
STYLE_IMAGE_FILE_NAME = 'style_image.png'
CHECKPOINT_FILE_NAME = 'epoch{:05d}.ckpt'
PICS_FILE_NAME = 'epoch{:05d}.png'
CHECKPOINT_BEST_FILE_NAME = 'best_model.ckpt'
GENERATED_IMG_FILE_NAME = 'transferred_image.png'

# 학습 / 테스트 모드
MODE_TRAIN = 'train'
MODE_TEST = 'test'

# 디바이스 종류
DEVICE_CUDA = 'cuda'
DEVICE_CPU = 'cpu'

# 하이퍼 파라미터
LEARNING_RATE = 0.01
BATCH_SIZE = 8
NUM_EPOCH = 300
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

# 옵션 값
SHUFFLE = True
TRACKING_FREQUENCY = 10
NUM_PICS_LIST = 10
NUM_RESULT_LIST = 10

# 그 외 기본 설정 값
RESIZE_SIZE = (120, 170)
LAMBDA_CONTENT = 1e1
LAMBDA_STYLE = 1e4

# state 저장시 딕셔너리 키 값
KEY_STATE_ORIGINAL_IMAGE = 'original_image'
KEY_STATE_OPTIMIZER = 'optimizer'
KEY_STATE_EPOCH = 'epoch'
KEY_STATE_SCORE = 'score'
# checkpoint_dict 딕셔너리 키 값
KEY_CHECKPOINT_FILE = 'checkpoint_file'
KEY_LEARNING_RATE = 'learning_rate'

# 초기 값
INITIAL_START_EPOCH_NUM = 1
INITIAL_BEST_ACCURACY_ZERO = 0
INITIAL_BEST_LOSS = 100000
