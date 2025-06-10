import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model.cnn_model import CNN
from attacks.fgsm import fgsm_attack

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Model setup
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# FGSM epsilon
epsilon = 0.25

# Adversarial Training
print("Adversarial training started...")
model.train()
for epoch in range(2):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()

        # Generate adversarial examples
        adv_images = fgsm_attack(model, images, labels, epsilon)

        # Combine clean + adversarial data
        combined_images = torch.cat([images, adv_images])
        combined_labels = torch.cat([labels, labels])

        # Train on combined data
        optimizer.zero_grad()
        combined_output = model(combined_images)
        combined_loss = criterion(combined_output, combined_labels)
        combined_loss.backward()
        optimizer.step()
print("Adversarial training complete.\n")

# Save adversarially trained model
torch.save(model.state_dict(), 'mnist_cnn_advtrained.pth')
print("Model saved to mnist_cnn_advtrained.pth")

# Evaluation function
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

# Evaluate on clean data
clean_acc = evaluate(model, test_loader)
print(f"[DEFENSE] Accuracy on clean test set: {clean_acc:.2f}%")

# Evaluate on adversarial data
print("[DEFENSE] Generating FGSM adversarial examples and evaluating...")
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
print(f"[DEFENSE] Accuracy on adversarial test set (epsilon={epsilon}): {adv_acc:.2f}%")

# --- Visualization of clean vs adversarial example ---
def show_adv_example(clean_img, adv_img, true_label, pred_clean, pred_adv):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(clean_img.squeeze().cpu(), cmap='gray')
    axes[0].set_title(f"Clean\nTrue: {true_label}, Pred: {pred_clean}")
    axes[1].imshow(adv_img.squeeze().cpu(), cmap='gray')
    axes[1].set_title(f"Adversarial\nPred: {pred_adv}")
    plt.tight_layout()
    plt.show()

# Pick a sample to visualize
sample_img, sample_label = test_dataset[1]
sample_tensor = sample_img.unsqueeze(0).to(device)
sample_tensor.requires_grad = True

# Clean prediction
clean_output = model(sample_tensor)
_, pred_clean = torch.max(clean_output.data, 1)

# FGSM on this sample
adv_tensor = fgsm_attack(model, sample_tensor, torch.tensor([sample_label]).to(device), epsilon)
adv_output = model(adv_tensor)
_, pred_adv = torch.max(adv_output.data, 1)

show_adv_example(sample_img, adv_tensor.squeeze().detach(), sample_label, pred_clean.item(), pred_adv.item())

# --- Inference block using trained model ---
inference_model = CNN().to(device)
inference_model.load_state_dict(torch.load('mnist_cnn_advtrained.pth'))
inference_model.eval()

with torch.no_grad():
    sample_img, sample_label = test_dataset[0]
    sample_img = sample_img.unsqueeze(0).to(device)
    output = inference_model(sample_img)
    _, predicted = torch.max(output, 1)
    print(f"[DEFENSE] Inference result: Predicted = {predicted.item()}, Actual = {sample_label}")