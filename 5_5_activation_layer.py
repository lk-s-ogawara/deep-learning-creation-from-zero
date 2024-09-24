import numpy as np

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0) # x は行列。対応する成分が0以下かそうでないかの行列を作る
        out = x.copy()
        out[self.mask] = 0 # mask行列がTrueの箇所のみ0で上書き

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

relu_layer = Relu()
x = np.array([[1.0, -0.5], [-2.0, 3.0]])
forward_out = relu_layer.forward(x)
backward_out = relu_layer.backward(forward_out)

print(forward_out)
print(backward_out)

class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out # 逆伝播の計算のために出力値を保持

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx
