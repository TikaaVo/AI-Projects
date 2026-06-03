""""
Pytorch CNN
Dataset: https://www.kaggle.com/datasets/alessiocorrado99/animals10
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
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
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
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
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.head(x)
        return x

if __name__ == '__main__':
    t = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    path = "Animals/raw-img"
    data = datasets.ImageFolder(path, transform=t)
    train_size = int(len(data) * 0.8)
    test_size = int(len(data) - train_size)

    train, test = random_split(data, [train_size, test_size])
    train = DataLoader(train, batch_size=32, shuffle=True)
    test = DataLoader(test, batch_size=32, shuffle=False)

    model = cnn().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)

    epochs = 30
    for epoch in range(epochs):
        loop = tqdm(train, desc=f"Epoch {epoch+1}/{epochs}")
        total = 0
        correct = 0
        model.train()
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

            loop.set_postfix(loss=loss.item(),acc=(correct/total)*100)

        model.eval()
        val_corr = 0
        val_tot = 0
        val_loss = 0
        with torch.no_grad():
            for features, labels in test:
                features, labels = features.to(device), labels.to(device)

                optimizer.zero_grad()
                val_preds = model(features)
                v_loss = criterion(val_preds, labels)
                val_loss += v_loss.item()

                _, val_preds = torch.max(val_preds, 1)
                val_tot += labels.size(0)
                val_corr += (val_preds == labels).sum().item()

        val_loss = val_loss / len(test)
        print(f"Validation accuracy: {(val_corr/val_tot)*100}, Validation loss: {val_loss}")
        scheduler.step(val_loss)