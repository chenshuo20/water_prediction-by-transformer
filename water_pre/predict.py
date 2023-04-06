import torch
import numpy as np
import models
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from train import split_data
import models
import matplotlib.pyplot as plt

model_path = './model.pth'
data_path = './Data.csv'
model = models.TransformerModel(4,4,2,128,512,0.1)
model.load_state_dict(torch.load(model_path))
model.eval()
df = pd.read_csv(data_path)
values = df.values
raw_data = torch.tensor(values, dtype=torch.float32)

data, label, maxs = split_data(data_path, 1000, 100, 12)
input_data = raw_data[-100:].unsqueeze(0)

predict = model(input_data)
predict = predict * maxs
output_data = predict.squeeze(0)
np.savetxt("output.txt", output_data[-12:].detach().numpy())

x = np.linspace(0, 100, 100)
fig, ax = plt.subplots(1,3,figsize=(18, 6))

ax[0].plot(x, output_data[:,1].detach().numpy(), label='IR')
ax[0].axvline(x=88, linestyle='--', color='gray', label='Today line')
ax[0].set_xlabel('Time/h')
ax[0].set_ylabel('Inflow Rate')
ax[0].set_title('The prediction')
plt.legend(loc='upper left')

ax[1].plot(x, output_data[:,1].detach().numpy(), label='CC')
ax[1].axvline(x=88, linestyle='--', color='gray', label='Today line')
ax[1].set_xlabel('Time/h')
ax[1].set_ylabel('COD concentration')
ax[1].set_title('The prediction')
plt.legend(loc='upper left')

ax[2].plot(x, output_data[:,1].detach().numpy(), label='ANC')
ax[2].axvline(x=88, linestyle='--', color='gray', label='Today line')
ax[2].set_xlabel('Time/h')
ax[2].set_ylabel('Ammonia Nitrogen Concentration')
ax[2].set_title('The prediction')

plt.legend(loc='upper left')
plt.savefig('output.png')
