'''
对数据进行分类，寻找最佳的系数
'''

from numpy import *
import matplotlib.pyplot as plt

'''
解析数据
'''


def loadDataSet(file_name):
    '''
        Desc:
            加载并解析数据
        Args:
            file_name -- 文件名称，要解析的文件所在磁盘位置
        Returns:
            dataMat -- 原始数据的特征
            labelMat -- 原始数据的标签，也就是每条样本对应的类别
        '''

    # dataMat为原始数据， labelMat为原始数据的标签

    dataMat = []
    labelMat = []

    fr = open(file_name)
    for line in fr.readlines():
        lineArr = line.strip().split()
        if len(lineArr) == 1:
            continue  # 这里如果就一个空的元素，则跳过本次循环
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))

    return dataMat, labelMat


# sigmoid函数
def sigmoid(X):
    return 1.0 / (1 + exp(-X))


# gradient descent
def gradAscent(dataMat, classLabels):
    '''
    Desc:
        正常的梯度上升法
    Args:
        dataMatIn -- 输入的 数据的特征 List
        classLabels -- 输入的数据的类别标签
    Returns:
        array(weights) -- 得到的最佳回归系数
    '''

    dataMatrix = mat(dataMat)  # 转换为矩阵(100 * 3)
    labelMat = mat(classLabels).transpose()  # (1 * 100)

    # m-样本数，n-特征数
    m, n = shape(dataMat)
    alpha = 0.001
    items = 500

    # 权值(3 * 1)
    weights = ones((n, 1))
    for k in range(items):
        h = sigmoid(dataMat * weights)  # (m * 1)
        error = (labelMat - h)  # 真实值 - 计算值
        weights = weights + alpha * dataMatrix.transpose() * error

    return array(weights)


'''
随机梯度下降
梯度下降优化算法在每次更新数据集时都需要遍历整个数据集，计算复杂都较高
随机梯度下降一次只用一个样本点来更新回归系数
'''


def stocGradAscent1(dataMatrix, classLabels,numiter = 150):
    '''
    Desc:
        随机梯度下降，只使用一个样本点来更新回归系数
    Args:
        dataMatrix -- 输入数据的数据特征（除去最后一列）
        classLabels -- 输入数据的类别标签（最后一列数据）
    Returns:
        weights -- 得到的最佳回归系数
    '''

    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numiter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.0001#alpha会随迭代不断减小，但是不会为0
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[dataIndex[randIndex]] * weights))
            error = classLabels[dataIndex[randIndex]] - h
            weights = weights + alpha * error * dataMatrix[dataIndex[randIndex]]
            del(dataIndex[randIndex])

        return weights


#可视化展示
def plotBestFit(dataArr, labelMat, weights):
    '''
        Desc:
            将我们得到的数据可视化展示出来
        Args:
            dataArr:样本数据的特征
            labelMat:样本数据的类别标签，即目标变量
            weights:回归系数
        Returns:
            None
    '''

    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    """
    y的由来，卧槽，是不是没看懂？
    首先理论上是这个样子的。
    dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
    w0*x0+w1*x1+w2*x2=f(x)
    x0最开始就设置为1叻， x2就是我们画图的y值，而f(x)被我们磨合误差给算到w0,w1,w2身上去了
    所以： w0+w1*x+w2*y=0 => y = (-w0-w1*x)/w2   
    """
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def simpleTest():
    # 1.收集并准备数据
    dataMat, labelMat = loadDataSet("TestSet.txt")

    # print dataMat, '---\n', labelMat
    # 2.训练模型，  f(x)=a1*x1+b2*x2+..+nn*xn中 (a1,b2, .., nn).T的矩阵值
    # 因为数组没有是复制n份， array的乘法就是乘法
    dataArr = array(dataMat)
    # print dataArr
    # weights = gradAscent(dataArr, labelMat)
    # weights = stocGradAscent0(dataArr, labelMat)
    weights = stocGradAscent1(dataArr, labelMat)
    # print '*'*30, weights

    # 数据可视化
    plotBestFit(dataArr, labelMat, weights)

    # --------------------------------------------------------------------------------
    # 从疝气病症预测病马的死亡率
    # 分类函数，根据回归系数和特征向量来计算 Sigmoid的值


def classifyVector(inX, weights):
    '''
    Desc:
        最终的分类函数，根据回归系数和特征向量来计算 Sigmoid 的值，大于0.5函数返回1，否则返回0
    Args:
        inX -- 特征向量，features
        weights -- 根据梯度下降/随机梯度下降 计算得到的回归系数
    Returns:
        如果 prob 计算大于 0.5 函数返回 1
        否则返回 0
    '''
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

    # 打开测试集和训练集,并对数据进行格式化处理


def colicTest():
    '''
    Desc:
        打开测试集和训练集，并对数据进行格式化处理
    Args:
        None
    Returns:
        errorRate -- 分类错误率
    '''
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    # 解析训练数据集中的数据特征和Labels
    # trainingSet 中存储训练数据集的特征，trainingLabels 存储训练数据集的样本对应的分类标签
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    # 使用 改进后的 随机梯度下降算法 求得在此数据集上的最佳回归系数 trainWeights
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    # trainWeights = stocGradAscent0(array(trainingSet), trainingLabels)
    errorCount = 0
    numTestVec = 0.0
    # 读取 测试数据集 进行测试，计算分类错误的样本条数和最终的错误率
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(
                currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)

    print("the error rate of this test is: %f" % errorRate)
    return errorRate

    # 调用 colicTest() 10次并求结果的平均值


def multiTest():
    numTests = 10
    errorSum = 0.0
    error = []
    for k in range(numTests):
        errorSum += colicTest()
        error.append(colicTest())
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))
    plt.plot(error)
    plt.show()

if __name__ == "__main__":
    #simpleTest()
    multiTest()