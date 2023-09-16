import pandas as pd
import torch

data = pd.read_csv('.\data\data.csv')

input = data.iloc[:, 0:2]
input = input.fillna(input.mean())
input = pd.get_dummies(input, dummy_na=True)
input_tensor = torch.Tensor(input.to_numpy(dtype=float))

print(input_tensor)
