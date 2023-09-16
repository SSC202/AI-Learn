import os

# 创建目录
os.makedirs(os.path.join('data'),exist_ok=True)
# 创建数据集文件
data_file = os.path.join('data','data.csv')
# 写入数据集文件
with open(data_file,'w') as f:
    f.write('NumRooms,Alley,Price\n')   # 列名
    f.write('NA,Pave,127500\n')         # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')