import numpy as np

def function_2(x):
    return x[0]**2 + x[1]**2

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h) の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 値を元に戻す
    
    return grad

print(numerical_gradient(function_2, np.array([3.0, 4.0])))
print(numerical_gradient(function_2, np.array([0.0, 2.0])))
print(numerical_gradient(function_2, np.array([3.0, 0.0])))

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x) # xにおける勾配を計算
        x -= lr * grad # 学習率*勾配 だけ線形に進んでxを更新

    return x

# 通常の例
init_x_0 = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x_0, lr=0.1, step_num=100))

# 学習率が大きすぎる例
init_x_1 = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x_1, lr=10.0, step_num=100))

# 学習率が小さすぎる例
init_x_2 = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x_2, lr=1e-10, step_num=100))
