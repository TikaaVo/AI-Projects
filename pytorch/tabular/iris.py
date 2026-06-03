"""
Pytorch NN
Dataset: https://www.kaggle.com/datasets/uciml/iris
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.label_map = {
            'Setosa': 0,
            'Versicolor': 1,
            'Virginica': 2
        }
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        features = row.iloc[:4]
        label = row.iloc[-1]

        features = torch.tensor(features.astype('float32').values, dtype=torch.float32)
        label = self.label_map[label]


        label = torch.tensor(label, dtype=torch.long)

        return features, label


class simpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 64)
        self.layer2 = nn.Linear(64, 3)
        pass

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x


data = CustomDataset("iris.csv")
train_size = int(0.8 * len(data))
test_size = len(data) - train_size

train_set, test_set = random_split(data, [train_size, test_size])

train = DataLoader(train_set, batch_size=32, shuffle=True)
test = DataLoader(test_set, batch_size=32, shuffle=False)

model = simpleNet()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    for features, labels in train:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for features, labels in test:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy: {:.2f}%".format(100 * correct / total))