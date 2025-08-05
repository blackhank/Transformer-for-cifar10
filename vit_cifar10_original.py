import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose, Resize
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from fuctions import plot_confusion_matrix
from torchvision.transforms import RandomCrop, RandomHorizontalFlip

class ViT(nn.Module):
    def __init__(self, image_size=32, patch_size=16, num_classes=10, dim=256, depth=8, heads=8, mlp_dim=512):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_conv = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True),
            num_layers=depth
        )
        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.patch_conv(img).flatten(2).transpose(1, 2)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(x)
        x = self.to_latent(x[:, 0])
        return self.linear_head(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_size = 32
patch_size = 16
num_classes = 10
dim = 256
depth = 8
heads = 8
mlp_dim = 512
epochs = 60
batch_size = 64
learning_rate = 1e-4

train_transform = Compose([
    Resize((image_size, image_size)),
    RandomCrop(image_size, padding=4),
    RandomHorizontalFlip(),
    ToTensor()
])
test_transform = Compose([
    Resize((image_size, image_size)),
    ToTensor()
])
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = ViT(
    image_size=image_size,
    patch_size=patch_size,
    num_classes=num_classes,
    dim=dim,
    depth=depth,
    heads=heads,
    mlp_dim=mlp_dim
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

train_losses = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
    scheduler.step()

plt.figure()
plt.plot(range(1, epochs + 1), train_losses, marker='o')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('loss_curve_cifar10_original.png')
plt.show()

print("Training finished and loss curve saved to loss_curve_cifar10_original.png")

model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=classes)
with open("classification_report_cifar10_original.txt", "w") as f:
    f.write(report)

plt.figure(figsize=(10, 8))
plot_confusion_matrix(cm, classes=classes, title='Confusion Matrix')
plt.savefig('confusion_matrix_cifar10_original.png')
plt.show()

torch.save(model.state_dict(), 'model_cifar10_original.pt')

print("Confusion matrix saved to confusion_matrix_cifar10_original.png")
print("Classification report saved to classification_report_cifar10_original.txt")
print("Model saved to model_cifar10_original.pt")