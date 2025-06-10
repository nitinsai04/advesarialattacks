import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model.cnn_model import CNN
from attacks.fgsm import fgsm_attack

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load adversarially trained model
model = CNN().to(device)
model.load_state_dict(torch.load('mnist_cnn_advtrained.pth'))
model.eval()

# Load MNIST test data
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Pick a test image
sample_img, sample_label = test_dataset[1]
sample_tensor = sample_img.unsqueeze(0).to(device)
sample_tensor.requires_grad = True

# Clean prediction
with torch.no_grad():
    clean_output = model(sample_tensor)
    _, pred_clean = torch.max(clean_output.data, 1)

# Generate adversarial image
epsilon = 0.25
adv_tensor = fgsm_attack(model, sample_tensor, torch.tensor([sample_label]).to(device), epsilon)

# Adversarial prediction
with torch.no_grad():
    adv_output = model(adv_tensor)
    _, pred_adv = torch.max(adv_output.data, 1)

# Output results
print(f"[DEFENSE] Actual Label     : {sample_label}")
print(f"[DEFENSE] Clean Prediction : {pred_clean.item()}")
print(f"[DEFENSE] Adversarial Pred : {pred_adv.item()}")

# Visual comparison
def show_images(clean_img, adv_img, clean_pred, adv_pred):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(clean_img.squeeze().cpu(), cmap='gray')
    axes[0].set_title(f"Clean\nPred: {clean_pred}")
    axes[1].imshow(adv_img.squeeze().cpu(), cmap='gray')
    axes[1].set_title(f"Adversarial\nPred: {adv_pred}")
    plt.tight_layout()
    plt.show()

show_images(sample_img, adv_tensor.squeeze().detach(), pred_clean.item(), pred_adv.item())