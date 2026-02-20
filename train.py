import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

# ================= CONFIG =================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 35
NUM_CLASSES = 4
MODEL_SAVE_PATH = "models/fracture_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================= TRANSFORMS =================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ================= DATASETS =================
train_dataset = datasets.ImageFolder("data/train", transform=transform)
val_dataset = datasets.ImageFolder("data/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print("Classes:", train_dataset.classes)

# ================= MODEL =================
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# ================= TRAINING LOOP =================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss:.4f}")

# ================= SAVE MODEL =================
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("✅ Model saved at", MODEL_SAVE_PATH)
