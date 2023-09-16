import torch

""" 张量创建 """
x = torch.arange(12)    # 创建张量
# print(x)
# print(x.shape)        # 求取张量（沿每个轴长度）的形状
# print(x.numel())      # 张量中元素的总数     
X = x.reshape(3,4)      # 改变张量形状
# print(X)
zero = torch.zeros((2,3,4)) # 创建零张量
# print(zero)
one = torch.ones((2,3,4))   # 创建单位张量
# print(one)
rand = torch.randn((3,4))   # 创建随机张量
# print(rand)

""" 张量的按元素运算 """
x1 = torch.tensor([1,2,3,4])
x2 = torch.tensor([5,6,7,8])
# print(x1 + x2)            # 按元素加
# print(x2 - x1)            # 按元素减
# print(x1 * x2)            # 按元素积
# print(x1 / x2)            # 按元素除
# print(x1 ** x2)           # 按元素求幂
  
""" 联结张量 """
x = torch.arange(12, dtype=torch.float32).reshape((3,4))
y = torch.tensor([[1,2,3,4],[1,2,3,4],[1,2,3,4]])
# print(torch.cat((x,y),dim = 0)) # 列方向联结
# print(torch.cat((x,y),dim = 1)) # 行方向联结

""" 关系运算符 """
z = (x == y)
# print(z)

""" 求全体元素的和 """
sum = x.sum()
# print(sum)

""" 广播机制 """
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a + b)    # a 会复制列，b 会复制行
