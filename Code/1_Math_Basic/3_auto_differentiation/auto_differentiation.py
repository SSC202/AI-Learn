import torch

"""
    当y为标量时
"""

x = torch.arange(4.0, requires_grad=True)

# y = 2*torch.dot(x, x)

# 反向传播算法计算y对x的梯度
# y.backward()
# x 具有grad(梯度)成员，使用zero_方法对其进行清零操作
# x.grad.zero_()

# y = x.sum()
# y.backward()
"""
    当y为向量时
"""
# 计算批量中每个样本的偏导数之和
# x.grad.zero_()
# y = x*x
# y.sum().backward()

"""
    将某一个变量从计算图中舍去
"""
y = x*x
u = y.detach()
z = x*u
z.sum().backward()

print(x.grad)
