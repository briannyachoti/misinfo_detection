import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import timm
import torch

class CIFakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label, category in enumerate(["fake", "real"]):
            category_path = os.path.join(root_dir, category)
            for img_name in os.listdir(category_path):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(category_path, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

def create_effnet_model():
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), device
