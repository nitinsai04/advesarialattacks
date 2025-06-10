import torch
import torch.nn.functional as F

def fgsm_attack(model, images, labels, epsilon):
    # Set requires_grad attribute of tensor. Important for attack
    images.requires_grad = True
    
    # Forward pass
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    
    # Zero all existing gradients
    model.zero_grad()
    
    # Backward pass
    loss.backward()
    
    # Collect the element-wise sign of the gradients
    data_grad = images.grad.data
    sign_data_grad = data_grad.sign()
    
    # Create perturbed image
    perturbed_image = images + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image