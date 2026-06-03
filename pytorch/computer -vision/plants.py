"""
Pytorch CNN
Dataset: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
"""

import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import random
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    
print(f"Using device {device}")

class cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(8*8*256, 38)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.head(x)
        return x

t = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

path = "/kaggle/input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"

train = datasets.ImageFolder(path + "/train", transform=t)
test = datasets.ImageFolder(path + "/valid", transform=t)

train = DataLoader(train, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
test = DataLoader(test, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

model = cnn()

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

epochs = 15
for epoch in range(epochs):
    loop = tqdm(train, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
    total = 0
    correct = 0
    for features, labels in loop:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()

        pred = model(features)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()

        _,pred = torch.max(pred, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

        loop.set_postfix(loss=loss.item(), acc=(correct/total)*100)



all_preds = []
all_labels = []
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for features, labels in test:
        features, labels = features.to(device), labels.to(device)
        preds = model(features)
        _, preds = torch.max(preds, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(20, 20))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

accuracy = (correct/total)*100
print(f"Test accuracy: {accuracy}")
torch.save(model.state_dict(), f"plant_disease_{int(accuracy)}.pth")

