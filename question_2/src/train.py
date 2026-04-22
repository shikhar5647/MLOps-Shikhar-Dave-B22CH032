import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataset import CityDataset, get_splits
from model import UNet
from metrics import compute_iou, dice_score

# paths
img_dir = "data/CameraRGB"
mask_dir = "data/CameraMask"

train_files, test_files = get_splits(img_dir)

transform = T.Compose([T.Resize((256, 512)), T.ToTensor()])

train_ds = CityDataset(img_dir, mask_dir, train_files, transform)
test_ds = CityDataset(img_dir, mask_dir, test_files, transform)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=4)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

epochs = 15

losses, ious, dices = [], [], []

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for img, mask in train_loader:
        img, mask = img.to(device), mask.to(device)

        pred = model(img)
        loss = criterion(pred, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # evaluation
    model.eval()
    iou, dice = 0, 0

    with torch.no_grad():
        for img, mask in test_loader:
            img, mask = img.to(device), mask.to(device)
            pred = model(img)

            iou += compute_iou(pred, mask)
            dice += dice_score(pred, mask)

    losses.append(total_loss)
    ious.append(iou / len(test_loader))
    dices.append(dice / len(test_loader))

    print(f"Epoch {epoch+1}: Loss={total_loss:.3f}, IoU={ious[-1]:.3f}, Dice={dices[-1]:.3f}")

# 🔹 Plot
plt.plot(losses, label="Loss")
plt.plot(ious, label="mIoU")
plt.plot(dices, label="Dice")
plt.legend()
plt.savefig("outputs/plots/training_curve.png")
