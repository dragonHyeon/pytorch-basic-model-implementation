import torch.nn.functional as F


def bce_loss(output, label):
    """
    * 해당 배치의 BCE loss 구하기
    :param output: shape: (batch, patch_size)
    :param label: shape: (batch, patch_size)
    :return: 해당하는 배치의 BCE loss
    """

    # BCE loss 계산
    batch_bce_loss = F.binary_cross_entropy(input=output,
                                            target=label).item()

    return batch_bce_loss


def l1_loss(fake_b, b):
    """
    * 해당 배치의 L1 loss 구하기
    :param fake_b: shape: (batch, SHAPE)
    :param b: shape: (batch, SHAPE)
    :return: 해당하는 배치의 L1 loss
    """

    # L1 loss 계산
    batch_l1_loss = F.l1_loss(input=fake_b,
                              target=b).item()

    return batch_l1_loss
