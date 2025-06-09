import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from model.cnn_model import CNN
from attacks.fgsm import fgsm_attack

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading and preprocessing
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Model setup
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
print("Training model...")
model.train()
for epoch in range(2):  # Keep epochs low for now
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
print("Training complete.\n")

# Evaluate on clean data
def evaluate(model, loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

clean_acc = evaluate(model, test_loader)
print(f"Accuracy on clean test set: {clean_acc:.2f}%")

# Evaluate on adversarial examples
print("Generating FGSM adversarial examples and evaluating...")
epsilon = 0.25
correct = 0
total = 0
model.eval()
for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    adv_images = fgsm_attack(model, images, labels, epsilon)
    outputs = model(adv_images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

adv_acc = 100 * correct / total
print(f"Accuracy on adversarial test set (epsilon={epsilon}): {adv_acc:.2f}%")