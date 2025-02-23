import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
from PIL import Image

class CIFAKE_Dataset(Dataset):
    """
    Custom PyTorch Dataset for CIFAKE (Real & AI-generated images).
    Assumes the dataset has 'train/real', 'train/fake', 'test/real', and 'test/fake' folders.
    """

    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = os.path.join(root_dir, split)  # 'train' or 'test'
        self.transform = transform
        self.data = []

        for label, category in enumerate(["real", "fake"]):  # 0 = Real, 1 = Fake
            class_dir = os.path.join(self.root_dir, category)
            for img_name in os.listdir(class_dir):
                self.data.append((os.path.join(class_dir, img_name), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

def load_cifake(data_path="../data/cifake_images_dataset", batch_size=32):
    """
    Loads the CIFAKE dataset and returns DataLoaders for training and testing.

    Args:
        data_path (str): Path to the CIFAKE dataset directory.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        train_loader (DataLoader): Training dataset loader.
        test_loader (DataLoader): Testing dataset loader.
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to CIFAR-10 size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = CIFAKE_Dataset(root_dir=data_path, split="train", transform=transform)
    test_dataset = CIFAKE_Dataset(root_dir=data_path, split="test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def show_cifake_images(data_path="../data/cifake_images_dataset", num_images=8):
    """
    Displays sample images from the CIFAKE dataset with their labels (Real or Fake).

    Args:
        data_path (str): Path to CIFAKE dataset.
        num_images (int): Number of images to display.
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=f"{data_path}/train", transform=transform)
    dataloader = DataLoader(dataset, batch_size=num_images, shuffle=True)

    images, labels = next(iter(dataloader))

    class_names = ["Real", "Fake"]

    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        image = images[i].permute(1, 2, 0).numpy()  
        image = (image - image.min()) / (image.max() - image.min())  

        axes[i].imshow(image)
        axes[i].set_title(f"Label: {class_names[labels[i].item()]}")
        axes[i].axis("off")

    plt.show()