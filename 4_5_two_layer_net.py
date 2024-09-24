import sys, os
import numpy as np
from dataset.mnist import load_mnist

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad

class TwoLayerNet:

    # 重みの初期化
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {} # ニューラルネットワークのパラメータを保持する
        # 1層目の重み（ランダムに初期化）
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        # 1層目のバイアス（ゼロで初期化）
        self.params['b1'] = np.zeros(hidden_size)
        # 2層目の重み（ランダムに初期化）
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        # 2層目のバイアス（ゼロで初期化）
        self.params['b2'] = np.zeros(output_size)

    # 推論
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1 # 入力値から1層目の結果を算出
        z1 = sigmoid(a1) # 1層目の結果にシグモイド関数を施す
        a2 = np.dot(z1, W2) + b2 # 2層目の結果を算出
        y = softmax(a2) # 出力結果にソフトマックス関数を施す

        return y

    # 損失関数（x:入力データ, t:教師データ）
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    # 認識精度の算出
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 重みパラメータに対する勾配
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

# 教師データと試験用データをMNISTから取得
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)


# ハイパーパラメータ
iters_num = 10000 # イテレーション回数
train_size = x_train.shape[0] # 入力データ
batch_size = 100 # ミニバッチの大きさ
learning_rate = 0.1 # 学習率

# 学習経過の記録用
train_loss_list = []

train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size / batch_size, 1) # 1エポックあたりの繰り返し数

# ネットワークの初期化
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size) # 入力データから無作為に100件取得

    # 訓練データのうち無作為に取得した100件をx_batch, t_batchに格納
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算
    grad = network.numerical_gradient(x_batch, t_batch)

    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key] # 学習率から勾配をかけた値を引く
    
    # 学習経過の記録（出力用）
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1エポックごとに認識制度を計算
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train) # 教師データの認識精度
        test_acc = network.accuracy(x_test, t_test) # 試験用データの認識精度
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
