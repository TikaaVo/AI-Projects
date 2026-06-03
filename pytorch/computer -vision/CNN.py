"""
General Pytorch CNN template
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
import pandas as pd
import os
from PIL import Image

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class submission(Dataset):
    def __init__(self, path, img_dir, transform):
        self.data = pd.read_csv(path)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        id = row.iloc[0]
        features = row.iloc[1]

        img_path = os.path.join(self.img_dir, features)
        img = Image.open(img_path).convert("L")

        if self.transform:
            img = self.transform(img)

        return img, id

class cnn(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
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
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64*6*6, 4)
        )

    def forward(self,x):
        x = self.conv(x)
        x = self.head(x)
        return x

t = transforms.Compose([
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

path = "train"
data = datasets.ImageFolder(path, t)

train_size = int(len(data) * 0.8)
test_size = int(len(data) - train_size)

train, test = random_split(data, [train_size, test_size])

train = DataLoader(train, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
test = DataLoader(test, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

model = cnn().to(device)
weights = torch.tensor([1.0, 1.0, 1.0, 15.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

epochs = 10
correct = 0
total = 0
for epoch in range(epochs):
    loop = tqdm(train, desc=f"Epoch {epoch+1}/{epochs}")
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

print(f"Test accuracy: {(correct/total)*100}")

submissions = submission("/submission/test.csv", "/submission/images", t)
submissions = DataLoader(submissions, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

model.eval()
predictions = []
with torch.no_grad():
    for features, ids in submissions:
        features = features.to(device)
        preds = model(features)
        _, preds = torch.max(preds, 1)
        preds = preds.cpu().tolist()
        predictions.extend(list(zip(ids, preds)))

df = pd.DataFrame(predictions, columns=['Id', 'Label'])
df.to_csv('submission.csv', index=False)