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

# Save the trained model weights
torch.save(model.state_dict(), 'mnist_cnn.pth')
print("Model saved to mnist_cnn.pth")

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

# --- Visualization of clean vs adversarial example ---
import matplotlib.pyplot as plt

# Visualize a sample clean vs adversarial image
def show_adv_example(clean_img, adv_img, true_label, pred_clean, pred_adv):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(clean_img.squeeze().cpu(), cmap='gray')
    axes[0].set_title(f"Clean\nTrue: {true_label}, Pred: {pred_clean}")
    axes[1].imshow(adv_img.squeeze().cpu(), cmap='gray')
    axes[1].set_title(f"Adversarial\nPred: {pred_adv}")
    plt.tight_layout()
    plt.show()

# Grab one test image and compare predictions
sample_img, sample_label = test_dataset[1]
sample_img_tensor = sample_img.unsqueeze(0).to(device)
sample_img_tensor.requires_grad = True

# Clean prediction
clean_output = model(sample_img_tensor)
_, pred_clean = torch.max(clean_output.data, 1)

# Generate adversarial example
adv_img_tensor = fgsm_attack(model, sample_img_tensor, torch.tensor([sample_label]).to(device), epsilon)
adv_output = model(adv_img_tensor)
_, pred_adv = torch.max(adv_output.data, 1)

# Visualize
show_adv_example(sample_img, adv_img_tensor.squeeze().detach(), sample_label, pred_clean.item(), pred_adv.item())

# Simple inference block
# Load model for inference
inference_model = CNN().to(device)
inference_model.load_state_dict(torch.load('mnist_cnn.pth'))
inference_model.eval()

# Run inference on one test sample
with torch.no_grad():
    sample_img, sample_label = test_dataset[0]
    sample_img = sample_img.unsqueeze(0).to(device)  # Add batch dimension
    output = inference_model(sample_img)
    _, predicted = torch.max(output, 1)
    print(f"Inference result: Predicted = {predicted.item()}, Actual = {sample_label}")