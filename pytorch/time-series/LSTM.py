"""
Pytorch LSTM
Dataset: Synthetic
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

time_steps = np.linspace(0, 500, 10000)
data = np.sin(time_steps)
data += np.random.normal(0, 0.1, size=data.shape)
data = data.astype(np.float32)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class dataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
    def __len__(self):
        return len(self.data) - self.window_size
    def __getitem__(self, idx):
        features = self.data[idx : idx + self.window_size]
        labels = self.data[idx + self.window_size]

        return torch.tensor(features), torch.tensor(labels)

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 50, batch_first=True)
        self.head = nn.Linear(50, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        x = self.head(last)
        return x

window_size = 50
datas = dataset(data, window_size)
train = DataLoader(datas, batch_size=32, shuffle=True)

model = LSTM().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

model.train()
epochs = 50
for epoch in range(epochs):
    loop = tqdm(train, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
    tot_loss = 0
    for features, labels in loop:
        features = features.unsqueeze(-1).to(device)
        labels = labels.unsqueeze(-1).to(device)

        optimizer.zero_grad()
        preds = model(features)
        loss = criterion(preds, labels)
        tot_loss += loss.item()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    tot_loss = tot_loss / len(loop)
    print(f"Loss: {tot_loss}")

model.eval()
test_input = data[-window_size:]
test_tensor = torch.tensor(test_input).view(1, window_size, 1).to(device)

with torch.no_grad():
    prediction = model(test_tensor).item()

print(f"True Next Value (Hypothetical): {np.sin(time_steps[-1] + (time_steps[1]-time_steps[0])):.4f}")
print(f"Model Prediction: {prediction:.4f}")