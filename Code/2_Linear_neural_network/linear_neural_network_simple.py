import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

true_w = torch.tensor([2, -3.4])
true_b = 2.8

# 创建数据集
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


def load_data(data_arrays, batch_size, is_Trained=True):
    """_summary_

    Args:
        data_arrays : 样本数据
        batch_size : 样本批量大小
        is_Trained : 是否在迭代周期内打乱数据

    Returns:
        _type_: _description_
    """
    # 构建数据迭代器
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_Trained)


batch_size = 10
# 获取迭代器并使用next获取第一项
data_iter = load_data((features, labels), batch_size)
# print(next(iter(data_iter)))

# 构建神经网络
net = nn.Sequential(nn.Linear(2, 1))
# 初始化模型参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
# 定义损失函数
loss = nn.MSELoss()
# 优化算法函数
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差:', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差:', true_b - b)