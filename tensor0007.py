import tensorflow as tf
import numpy as np
import tensormodel as nn

# XOR 논리 게이트 구현.
## Relu 함수 + softmax entropy
## Relu 함수의 오차는 빠르게 줄기 때문에, 오차만 놓고 볼 때 일정 회수 (100) 이상의 반복 학습은 의미가 없다.
## OR, AND 구현에서도 확인할 수 있지만,
## Relu 함수를 적용한 논리 게이트의 학습 결과는 종종 오류가 발생하기 때문에
## 논리 게이트의 학습에 잘 맞지 않는다.

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

X = nn.DefPlaceHolder(2, 'Input')
Y = nn.DefPlaceHolder(2, 'Target')

hidden = nn.NnFullConnectedLayer(X, 2, 2, nn.relu_tf, bias='Variable')
output = nn.NnFullConnectedLayer(hidden.activation, 2, 2, nn.relu_tf, bias='Variable')
learn = nn.SoftmaxCrossEntropy(Y, output)

run = nn.RunLearnModel("XOR with Relu (softmax)")
run.Learn(learn, feed_dict={X:x_data, Y:y_data})
run.Test_2(Y, output, feed_dict={X:x_data, Y:y_data})
