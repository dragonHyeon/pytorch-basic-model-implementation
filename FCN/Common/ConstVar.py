import os

# 파일 경로
PROJECT_ROOT_DIRECTORY = os.path.dirname(os.getcwd())
TRAIN_DATA_DIR_X = '{0}/RES/chest-xray-masks-and-labels/data/Lung Segmentation/yh_modified_dir/train/CXR_png/'.format(PROJECT_ROOT_DIRECTORY)
TRAIN_DATA_DIR_Y = '{0}/RES/chest-xray-masks-and-labels/data/Lung Segmentation/yh_modified_dir/train/masks/'.format(PROJECT_ROOT_DIRECTORY)
TEST_DATA_DIR_X = '{0}/RES/chest-xray-masks-and-labels/data/Lung Segmentation/yh_modified_dir/test/CXR_png/'.format(PROJECT_ROOT_DIRECTORY)
TEST_DATA_DIR_Y = '{0}/RES/chest-xray-masks-and-labels/data/Lung Segmentation/yh_modified_dir/test/masks/'.format(PROJECT_ROOT_DIRECTORY)
OUTPUT_DIR = '{0}/DATA/'.format(PROJECT_ROOT_DIRECTORY)
OUTPUT_DIR_SUFFIX_CHECKPOINT = 'checkpoint'
OUTPUT_DIR_SUFFIX_PICS = 'pics'
OUTPUT_DIR_SUFFIX_SEGMENTATION_IMG = 'segmentation_img/{0}'
OUTPUT_DIR_SUFFIX_ORIGINAL = 'original'
OUTPUT_DIR_SUFFIX_SEGMENTED = 'segmented'
INPUT_DIR = '{0}/RES/sample/'.format(PROJECT_ROOT_DIRECTORY)
CHECKPOINT_FILE_NAME = 'epoch{:05d}.ckpt'
PICS_FILE_NAME = 'epoch{:05d}.png'
CHECKPOINT_BEST_FILE_NAME = 'best_model.ckpt'
SEGMENTATION_IMG_FILE_NAME = '{0}.png'

# 학습 / 테스트 모드
MODE_TRAIN = 'train'
MODE_TEST = 'test'

# 여러 키 값
KEY_X = 'x'
KEY_Y = 'y'

# 디바이스 종류
DEVICE_CUDA = 'cuda'
DEVICE_CPU = 'cpu'

# 하이퍼 파라미터
LEARNING_RATE = 0.001
BATCH_SIZE = 8
NUM_EPOCH = 20

# 옵션 값
SHUFFLE = True
TRACKING_FREQUENCY = 1
NUM_PICS_LIST = 5

# 그 외 기본 설정 값
NUM_CLASSES = 2
RESIZE_SIZE = 224

# state 저장시 딕셔너리 키 값
KEY_STATE_MODEL = 'model'
KEY_STATE_OPTIMIZER = 'optimizer'
KEY_STATE_EPOCH = 'epoch'
KEY_STATE_SCORE = 'score'

# 초기 값
INITIAL_START_EPOCH_NUM = 1
INITIAL_BEST_MIOU_ZERO = 0
