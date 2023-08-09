import torch.nn as nn

loss_fn = nn.CrossEntropyLoss()


"""
nn.CrossEntropyLoss()
def forward(self, input: Tensor, target: Tensor) -> Tensor:

사용 설명
batch size = 2 인 경우
input(예측한 값): model 의 output 값. ex) [[-0.3588, -0.0903,  0.0114], [-0.3502, -0.0834,  0.0395]]
target(실제 값): label 값. ex) [0, 2]

원래는 softmax 된 배열이어야 하지만
nn.CrossEntropyLoss() 에서는 내부적으로
softmax 를 수행해주기 때문에 그냥 input 을
그대로 넣어주면 됨.

loss_fn = nn.CrossEntropyLoss()
loss_fn(input, target)
"""
