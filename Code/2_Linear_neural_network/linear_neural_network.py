import random
import torch
import d2l.torch as d2l


def synthetic_data(w, b, num_examples):
    """生成数据集，满足正态分布

    Args:
        w : 权重向量
        b : 偏置
        num_examples : 样本的个数
    Returns:
        元组,第一个元素为特征张量,第二个元素为标签张量
    """
    X = torch.normal(0, 1, (num_examples, len(w)))
    # y = Wx + b
    y = torch.matmul(X, w)+b
    # 加入高斯噪声项
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


# 生成模拟数据,自然生成列向量
tr_w = torch.tensor([2, -3.4])
tr_b = 4.2
# 生成特征和标签值
features, labels = synthetic_data(tr_w, tr_b, 1000)
# 生成散点图
# d2l.set_figsize()
# d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
# d2l.plt.show()


def data_iter(batch_size, features, labels):
    """随机小批量读取数据集

    Args:
        batch_size: 样本批量大小
        features: 数据集特征矩阵
        labels: 标签向量
    Yields:
        随机样本特征子矩阵和标签子矩阵(生成器)
    """
    # batch:批
    num_examples = len(features)
    # 生成0 - num_examples 数据的索引列表
    indices = list(range(num_examples))     # indices:指标
    # 随机打乱列表（洗牌函数）
    random.shuffle(indices)                 # shuffle:洗牌
    # 将打乱后的数组分割，每段为batch_size，输出时使用生成器进行伪随机输出
    # 每次调用生成器时，batch_indices中的值会进行更新
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i:min(num_examples, i+batch_size)])
        yield features[batch_indices], labels[batch_indices]


batch_size = 10


# for X, y in data_iter(batch_size, features, labels):
#    print(X, '\n', y)
#    break


# 初始化模型参数,requires_grad表示表示需要计算梯度
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


def linreg(X, w, b):
    """线性回归模型函数

    Args:
        X : 小批量样本特征矩阵 
        w : 权重向量 
        b : 偏置 
    Returns:
        求得的预测标签
    """
    return torch.matmul(X, w)+b


def squared_loss(y, y_hat):
    """平方损失函数

    Args:
        y : 实际标签向量
        y_hat : 预测标签向量
    Returns:
        平方损失
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    """随机梯度下降函数

    Args:
        params : 神经网络参数
        lr : 学习率
        batch_size : 批量大小
    """
    # 禁用梯度计算
    with torch.no_grad():
        for param in params:
            # 进行梯度下降计算
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# 超参数定义
lr = 0.03       # 学习率
num_epochs = 4  # 迭代周期个数
# 神经网络和损失函数定义
net = linreg
loss = squared_loss
# 训练
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):   # 取小批量的样本
        l = loss(y, net(X, w, b))                          # 模型输出并计算损失函数
        l.sum().backward()                                 # 先求和，再求梯度
        sgd([w, b], lr, batch_size)                        # 随机梯度下降
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)                     # 将求出的参数代入原模型，计算损失函数
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

with torch.no_grad():
    print(w)
    print(b)