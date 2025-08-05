import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose, Resize, ColorJitter, RandomRotation
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from fuctions import plot_confusion_matrix
from torchvision.transforms import RandomCrop, RandomHorizontalFlip

# Define the Vision Transformer (ViT) model
class ViT(nn.Module):
    def __init__(self, image_size=32, patch_size=16, num_classes=10, dim=256, depth=8, heads=8, mlp_dim=512):
        super().__init__()
        # Calculate the number of patches
        num_patches = (image_size // patch_size) ** 2
        # Patch embedding: Conv2d to project image patches into a higher dimension
        self.patch_conv = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        # Positional embedding: Learnable parameters to encode patch positions
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # Class token: A learnable parameter prepended to the sequence of patch embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # Transformer Encoder: Composed of multiple encoder layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True),
            num_layers=depth
        )
        # To latent: Identity layer, can be replaced with a more complex projection if needed
        self.to_latent = nn.Identity()
        # Linear head: Final classification layer
        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        # 1. Patch embedding:
        # Apply convolution to extract patches, then flatten and transpose
        # img: (batch_size, channels, height, width) -> patch_conv: (batch_size, dim, num_patches_h, num_patches_w)
        # flatten(2): (batch_size, dim, num_patches) -> transpose(1, 2): (batch_size, num_patches, dim)
        x = self.patch_conv(img).flatten(2).transpose(1, 2)
        b, n, _ = x.shape # b: batch_size, n: number of patches

        # 2. Add class token:
        # Expand class token to match batch size
        cls_tokens = self.cls_token.expand(b, -1, -1)
        # Concatenate class token with patch embeddings
        x = torch.cat((cls_tokens, x), dim=1)

        # 3. Add positional embedding:
        # Add learnable positional embeddings to the combined sequence
        x += self.pos_embedding[:, :(n + 1)]

        # 4. Pass through Transformer Encoder:
        x = self.transformer(x)

        # 5. Classification:
        # Take the output corresponding to the class token (first element in the sequence)
        x = self.to_latent(x[:, 0])
        # Pass through the linear classification head
        return self.linear_head(x)

# Set device for training (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters for the ViT model and training
image_size = 32  # Input image size (CIFAR-10 images are 32x32)
patch_size = 16  # Size of each image patch
num_classes = 10  # Number of output classes (CIFAR-10 has 10 classes)
dim = 512  # Dimension of the patch embeddings and transformer
depth = 12  # Number of transformer encoder layers
heads = 16  # Number of attention heads in each transformer layer
mlp_dim = 1024  # Dimension of the Multi-Layer Perceptron (MLP) in transformer
epochs = 150  # Number of training epochs
batch_size = 64  # Batch size for training and evaluation
learning_rate = 1e-4  # Learning rate for the optimizer

# Load CIFAR-10 dataset
# Define transformations for training and testing datasets
train_transform = Compose([
    Resize((image_size, image_size)),  # Resize images to the specified size
    RandomCrop(image_size, padding=4),  # Randomly crop the image with padding
    RandomHorizontalFlip(),  # Randomly flip the image horizontally
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Randomly change brightness, contrast, saturation, and hue
    RandomRotation(10), # Randomly rotate the image by up to 10 degrees
    ToTensor()  # Convert images to PyTorch tensors
])
test_transform = Compose([
    Resize((image_size, image_size)),  # Resize images to the specified size
    ToTensor()  # Convert images to PyTorch tensors
])

# Load CIFAR-10 datasets with defined transformations
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)

# Create data loaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = ViT(
    image_size=image_size,
    patch_size=patch_size,
    num_classes=num_classes,
    dim=dim,
    depth=depth,
    heads=heads,
    mlp_dim=mlp_dim
).to(device) # Move model to the specified device (GPU/CPU)

criterion = nn.CrossEntropyLoss() # Cross-entropy loss for classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam optimizer
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) # Cosine annealing learning rate scheduler

# Training loop
train_losses = [] # List to store training losses per epoch
for epoch in range(epochs):
    model.train() # Set model to training mode
    running_loss = 0.0 # Initialize running loss for the current epoch
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device) # Move data to device
        optimizer.zero_grad() # Zero the gradients
        outputs = model(images) # Forward pass
        loss = criterion(outputs, labels) # Calculate loss
        loss.backward() # Backward pass (compute gradients)
        optimizer.step() # Update model parameters
        running_loss += loss.item() # Accumulate loss
    epoch_loss = running_loss / len(train_loader) # Calculate average loss for the epoch
    train_losses.append(epoch_loss) # Store epoch loss
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}") # Print epoch loss
    scheduler.step() # Update learning rate scheduler

# Plotting the loss curve
plt.figure() # Create a new figure
plt.plot(range(1, epochs + 1), train_losses, marker='o') # Plot training losses
plt.title('Training Loss Over Epochs') # Set plot title
plt.xlabel('Epochs') # Set x-axis label
plt.ylabel('Loss') # Set y-axis label
plt.grid(True) # Add grid
plt.savefig('loss_curve_cifar10.png') # Save the plot to a file
plt.show() # Display the plot

print("Training finished and loss curve saved to loss_curve_cifar10.png")

# Evaluation and Confusion Matrix
model.eval() # Set model to evaluation mode
all_preds = [] # List to store all predictions
all_labels = [] # List to store all true labels
with torch.no_grad(): # Disable gradient calculation during evaluation
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device) # Move data to device
        outputs = model(images) # Forward pass
        _, predicted = torch.max(outputs.data, 1) # Get predicted classes
        all_preds.extend(predicted.cpu().numpy()) # Store predictions
        all_labels.extend(labels.cpu().numpy()) # Store true labels

# Define class names for CIFAR-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# Calculate confusion matrix
cm = confusion_matrix(all_labels, all_preds)
# Generate classification report
report = classification_report(all_labels, all_preds, target_names=classes)
# Save classification report to a text file
with open("classification_report_cifar10.txt", "w") as f:
    f.write(report)

# Plotting the confusion matrix
plt.figure(figsize=(10, 8)) # Create a new figure with specified size
plot_confusion_matrix(cm, classes=classes, title='Confusion Matrix') # Plot confusion matrix
plt.savefig('confusion_matrix_cifar10.png') # Save the plot to a file
plt.show() # Display the plot

# Save the trained model
torch.save(model.state_dict(), 'model_cifar10.pt')

print("Confusion matrix saved to confusion_matrix_cifar10.png")
print("Classification report saved to classification_report_cifar10.txt")
print("Model saved to model_cifar10.pt")