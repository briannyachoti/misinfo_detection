import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

class CIFAKE_Dataset(Dataset):
    """
    Custom PyTorch Dataset for CIFAKE (Real & AI-generated images).
    Assumes the dataset has two folders: 'real/' and 'fake/'.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        # Load real images
        real_dir = os.path.join(root_dir, "real")
        for img_name in os.listdir(real_dir):
            self.data.append((os.path.join(real_dir, img_name), 0))  # 0 for real

        # Load fake images
        fake_dir = os.path.join(root_dir, "fake")
        for img_name in os.listdir(fake_dir):
            self.data.append((os.path.join(fake_dir, img_name), 1))  # 1 for fake

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# Function to load CIFAKE dataset
def load_cifake(batch_size=32, data_path="./CIFAKE"):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to CIFAR-10 size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = CIFAKE_Dataset(root_dir=data_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    return data_loader



