"""
Pytorch CNN
Dataset: https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset
"""

import torch
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from tqdm import tqdm
import os
from PIL import Image
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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class dataset(Dataset):
    def __init__(self, path, img_dir, transform):
        self.data = pd.read_csv(path)
        self.img_dir = img_dir
        self.transform = transform
        self.classes = sorted(self.data.iloc[:, 1].unique())
        self.label_map = {label: i for i, label in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        feature = row.iloc[0]
        label = row.iloc[1]

        img_path = os.path.join(self.img_dir, feature)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.label_map[label]
        label = torch.tensor(label, dtype=torch.long)

        return image, label

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
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(5*5*256, 31)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.head(x)
        return x


set_seed(42)

t_train = transforms.Compose([
    transforms.Resize((160, 160)),

    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

t_test = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

img_dir = "/kaggle/input/face-recognition-dataset/Faces/Faces"
csv_path = "/kaggle/input/face-recognition-dataset/Dataset.csv"

train = dataset(csv_path, img_dir, t_train)
test = dataset(csv_path, img_dir, t_test)

train_size = int(0.8 * len(data))
indices = torch.randperm(len(train)).tolist()
train_idx = indices[:train_size]
val_idx = indices[train_size:]

train = Subset(train, train_idx)
test = Subset(test, val_idx)

train = DataLoader(train, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
test = DataLoader(test, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

model = cnn()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 20
for epoch in range(epochs):
    loop = tqdm(train, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
    total = 0
    correct = 0
    for features, labels in loop:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()

        preds = model(features)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(preds, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

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
torch.save(model.state_dict(), f"face_recognition_{int(accuracy)}.pth")