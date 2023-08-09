import torch.nn.functional as F

from DeepLearning import utils


def content_loss(original_features, content_features, layer_name='conv5_1'):
    """
    * Content loss 계산 및 반환
    :param original_features: 원본 이미지 features
    :param content_features: 유지할 이미지 features
    :param layer_name: 해당 깊이의 layer 이름
    :return: Content loss
    """

    return F.mse_loss(input=original_features[layer_name], target=content_features[layer_name])


class StyleLoss:
    def __init__(self, loss_lambda_dict={'conv1_1':0.75, 'conv2_1':0.5, 'conv3_1':0.25, 'conv4_1':0.25, 'conv5_1':0.25}):
        """
        * Style loss
        :param loss_lambda_dict: 깊이에 따른 loss 가중치 값 담은 dict
        """

        # 깊이에 따른 loss 가중치 값 담은 dict
        self.loss_lambda_dict = loss_lambda_dict

    def __call__(self, original_features, style_features):
        """
        * Style loss 계산 및 반환
        :param original_features: 원본 이미지 features
        :param style_features: 스타일 이미지 features
        :return: Style loss
        """

        loss = 0
        for layer_name in self.loss_lambda_dict:

            # 원본 이미지 gram matrix
            original_gram_matrix = utils.get_gram_matrix(feature=original_features[layer_name])
            # 스타일 이미지 gram matrix
            style_gram_matrix = utils.get_gram_matrix(feature=style_features[layer_name])

            # Style loss 계산
            n, c, h, w = original_features[layer_name].shape
            loss += self.loss_lambda_dict[layer_name] * F.mse_loss(input=original_gram_matrix, target=style_gram_matrix) / (n * c * h * w)

        return loss
