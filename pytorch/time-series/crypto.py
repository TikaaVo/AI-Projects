"""
Pytorch LSTM
Dataset: https://www.kaggle.com/datasets/mihikaajayjadhav/top-100-cryptocurrencies-daily-price-data-2025
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using {device}")

class train_dataset(Dataset):
    def __init__(self, data, window_size, target):
        self.data = data
        self.window_size = window_size
        self.target = target
    def __len__(self):
        return len(self.data) - self.window_size
    def __getitem__(self,idx):
        features = self.data[idx : idx + self.window_size]
        labels = self.data[idx + self.window_size, self.target]

        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

class test_dataset(Dataset):
    def __init__(self, data, window_size, ids):
        self.data = data
        self.ids = ids
        self.window_size = window_size
    def __len__(self):
        return len(self.data) - self.window_size
    def __getitem__(self,idx):
        features = self.data[idx : idx + self.window_size]
        id = self.ids[idx + self.window_size]

        return torch.tensor(features, dtype=torch.float32), id

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(8, 50, batch_first=True)
        self.head = nn.Linear(50, 1)
    def forward(self,x):
        x, _ = self.lstm(x)
        x = self.head(x[:, -1, :])
        return x

window_size = 50

train = pd.read_csv('crypto/crypto_historical_365days.csv').dropna()
#test = pd.read_csv('crypto/test.csv').dropna()

selected = ['price', 'market_cap', 'volume', 'daily_return', 'price_ma7',
            'price_ma30', 'volatility_7d', 'cumulative_return']
train = train[selected]
#ids = test['id'].values
#test = test[selected]

idx = int(len(train) * 0.8)
val = train[idx:]
train = train[:idx]

target = train.columns.get_loc('price')

scaler = MinMaxScaler()
lab_scaler = MinMaxScaler()

scaler.fit(train)
lab_scaler.fit(train['price'].values.reshape(-1, 1))

train = scaler.transform(train)
val = scaler.transform(val)
#test = scaler.transform(test)

train = train_dataset(train, window_size, target)
val = train_dataset(val, window_size, target)
#test = test_dataset(test, window_size, ids)

train = DataLoader(train, batch_size=32, shuffle=True)
val = DataLoader(val, batch_size=32, shuffle=False)
#test = DataLoader(test, batch_size=32, shuffle=False)

model = LSTM().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

epochs = 50
for epoch in range(epochs):
    loop = tqdm(train, desc=f"Epoch {epoch+1}/{epochs}")
    model.train()
    for features, labels in loop:
        features, labels = features.to(device), labels.unsqueeze(-1).to(device)

        optimizer.zero_grad()
        preds = model(features)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    model.eval()
    tot_loss = 0
    tot_diff = 0
    with torch.no_grad():
        for features, labels in val:
            features, labels = features.to(device), labels.unsqueeze(-1).to(device)
            preds = model(features)
            loss = criterion(preds, labels)
            tot_loss += loss.item()

            real_pred = lab_scaler.inverse_transform(preds.cpu().numpy())
            real_label = lab_scaler.inverse_transform(labels.cpu().numpy())

            error_pct = np.abs((real_label - real_pred) / (real_label + 1e-8)) * 100
            tot_diff += np.mean(error_pct)

    tot_loss = tot_loss / len(val)
    scheduler.step(tot_loss)
    print(f"Validation loss: {tot_loss}")
    tot_diff = tot_diff / len(val)
    print(f"Average Error: {tot_diff:.2f}%")

#model.eval()
#tot_loss = 0
#final = []
#id_list = []
#with torch.no_grad():
    #for features, ids in test:
        #features = features.to(device)
        #preds = model(features)
        #final.extend(x.item() for x in preds)
        #id_list.extend(ids)

#final = np.array(final)
#final = lab_scaler.inverse_transform(final.reshape(-1, 1))
#final = final.flatten()

#submission = pd.DataFrame({
    #'id': id_list,
    #'price': final
#})

#submission.to_csv('submission.csv', index=False)