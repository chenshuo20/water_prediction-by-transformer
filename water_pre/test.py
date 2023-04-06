import pickle
import pandas as pd
import torch

path = './Data.csv'
df = pd.read_csv(path)

values = df.values
data = torch.tensor(values, dtype=torch.float32)

print(data.size()[0])