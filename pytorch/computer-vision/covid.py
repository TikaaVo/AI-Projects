"""
Pytorch CNN
Dataset: https://www.kaggle.com/datasets/kaggleprollc/covid-19-image-data-collection-ieee
"""

from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import transforms, datasets
import torch.nn as nn
import torch
from tqdm import tqdm
import os
from PIL import Image
import pandas as pd

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.convo = nn.Sequential(
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
        self.final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*7*7, 3),
        )
    def forward(self,x):
        x = self.convo(x)
        x = self.final(x)
        return x

class inp(Dataset):
    def __init__(self, path, img_dir, transform):
        self.data = pd.read_csv(path)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        feature = row.iloc[0]
        label = row.iloc[1]

        img_path = os.path.join(self.img_dir, feature)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(label, dtype=torch.long)

        return img, label

class out(Dataset):
    def __init__(self, img_dir, transform):
        self.data = [f for f in os.listdir(img_dir) if f.endswith('jpg')]
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        image_name = self.data[idx]
        path = os.path.join(self.img_dir, image_name)
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, image_name


t = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

t_final = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train = datasets.ImageFolder("COVID_IEEE/train", transform=t)
val   = datasets.ImageFolder("COVID_IEEE/train", transform=t_final)
num_train = int(len(train) * 0.8)
indices = torch.randperm(len(train)).tolist()

train_idx = indices[:num_train]
val_idx   = indices[num_train:]

train_data = Subset(train, train_idx)
test_data  = Subset(val, val_idx)

train = DataLoader(train_data, batch_size=32, shuffle=True)
test  = DataLoader(test_data, batch_size=32, shuffle=False)

final = out("COVID_IEEE/test", t_final)
final = DataLoader(final, batch_size=32, shuffle=False)

model = conv().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 50
for epoch in range(epochs):
    loop = tqdm(train, leave=True)
    loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
    epoch_loss = 0
    correct = 0
    total = 0
    for features, labels in loop:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        pred = model(features)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(pred, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loop.set_postfix(loss=loss.item(), acc=f"{100 * correct / total:.2f}%")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for features, labels in test:
        features, labels = features.to(device), labels.to(device)
        pred = model(features)
        _, predicted = torch.max(pred, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy: {:.2f}%".format(100 * correct / total))

model.eval()
results = []
with torch.no_grad():
    for features, idxs in final:
        features = features.to(device)
        pred = model(features)
        _, predicted = torch.max(pred, 1)
        predicted = predicted.cpu().tolist()
        results.extend(list(zip(idxs, predicted)))

df = pd.DataFrame(results, columns=["filename", "label"])
df.to_csv("submission.csv", index=False)