import numpy as np

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T) # Tは転置
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None    # 損失
        self.y = None       # softmaxの出力
        self.t = None       # 教師データ(one-hot vector)
    
    def forward(self, x, t):
        self.t = t # tは教師データ
        self.y = softmax(x) # softmaxは活性化関数
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1): # 出力レイヤなのでdoutは固定
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size # 正規化誤差が逆伝播する

        return dx

