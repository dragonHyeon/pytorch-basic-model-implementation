from torchvision import transforms
from torchvision.datasets import CIFAR100

from Common import ConstVar

# CIFAR-100 학습용 데이터셋
CIFAR100_train = CIFAR100(root=ConstVar.DATA_DIR_TRAIN,
                          train=True,
                          transform=transforms.Compose([
                              transforms.Resize(size=ConstVar.RESIZE_SIZE),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=ConstVar.NORMALIZE_MEAN, std=ConstVar.NORMALIZE_STD)
                          ]),
                          download=True)

# CIFAR-100 테스트용 데이터셋
CIFAR100_test = CIFAR100(root=ConstVar.DATA_DIR_TEST,
                         train=False,
                         transform=transforms.Compose([
                             transforms.Resize(size=ConstVar.RESIZE_SIZE),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=ConstVar.NORMALIZE_MEAN, std=ConstVar.NORMALIZE_STD)
                         ]),
                         download=True)
