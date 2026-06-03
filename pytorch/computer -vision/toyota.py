"""
Pytorch CNN
Dataset: https://www.kaggle.com/datasets/stevenalbert15/toyota-corolla-car-parts
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

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

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 7),
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.head(x)
        return x


t_train = transforms.Compose([
    transforms.Resize((512, 640)),

    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.25, contrast=0.33),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5])
])

t_test = transforms.Compose([
    transforms.Resize((512, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5])
])

path = "Toyota Corolla Dataset"
train = datasets.ImageFolder(path, t_train)
test = datasets.ImageFolder(path, t_test)

train_size = int(0.8 * len(train))
indices = torch.randperm(len(train)).tolist()
train_idx = indices[:train_size]
test_idx = indices[train_size:]

train = Subset(train, train_idx)
test = Subset(test, test_idx)

train = DataLoader(train, batch_size=16, shuffle=True)
test = DataLoader(test, batch_size=16, shuffle=False)

model = cnn().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.03)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=2
)

epochs = 10
for epoch in range(epochs):
    loop = tqdm(train, desc=f"Epoch {epoch+1}/{epochs}")
    model.train()
    correct = 0
    total = 0
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

    correct_val = 0
    total_val = 0
    loss_val = 0
    model.eval()
    with torch.no_grad():
        for features, labels in test:
            features, labels = features.to(device), labels.to(device)

            preds = model(features)
            loss = criterion(preds, labels)
            loss_val += loss.item()
            _, preds = torch.max(preds, 1)
            total_val += labels.size(0)
            correct_val += (preds == labels).sum().item()

    avgloss_val = loss_val / len(test)
    print(f"Epoch {epoch+1} complete. Validation loss: {avgloss_val}, Validation Accuracy: {(correct_val / total_val) * 100}")

    scheduler.step(avgloss_val)