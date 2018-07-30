import numpy as np

#反向传播算法

def nonlin(x, deri=False):
    if (deri == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


x = np.array([[0.35], [0.9]])  # 输入值
y = np.array([[0.5]])  # 输出值

np.random.seed(1)

w0 = np.array([[0.1, 0.8], [0.4, 0.6]])  # 第一层权值
w1 = np.array([[0.3, 0.9]])  # 第二层权值

print('original', w0, '\n', w1)

for j in range(100):  # 迭代100次
    l0 = x
    l1 = nonlin(np.dot(w0, l0))  # w0*l0
    l2 = nonlin(np.dot(w1, l1))
    l2_error = y - l2
    Error = 1 / 2.0 * (y - l2) ** 2
    print('Error', Error)

    l2_delta = l2_error * nonlin(l2, deri=True)

    l1_error = l2_delta * w1
    l1_delta = l1_error * nonlin(l1, deri=True)

    w1 += l2_delta * l1.T
    w0 += l0.T.dot(l1_delta)
    print(w0, '\n', w1)
