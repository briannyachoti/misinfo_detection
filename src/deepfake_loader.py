import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# Define dataset path
data_dir = "../data/deepfake_detection_dataset"

# Define transformations (resize, normalize, convert to tensor)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to a fixed size
    transforms.ToTensor()  # Convert images to PyTorch tensors
])

# Load dataset using ImageFolder
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Save the dataset using pickle for fast reloading
import pickle
with open("deepfake_dataset.pkl", "wb") as f:
    pickle.dump(dataset, f)

print(f"Loaded {len(dataset)} images successfully!")
