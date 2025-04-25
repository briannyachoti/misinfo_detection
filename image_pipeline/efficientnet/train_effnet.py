import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import timm
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import copy

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    TRAIN_IMG_DIR, TEST_IMG_DIR, EFFNET_OUTPUT, EFFNET_MODEL_PATH,
    EFFNET_PRED_CSV, EFFNET_CM_PNG, EFFNET_ROC_PNG, EFFNET_UMAP_PNG, EFFNET_TSNE_PNG
)

# Enable GPU optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

class CIFakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        for label, category in enumerate(["fake", "real"]):
            category_path = os.path.join(root_dir, category)
            for img_name in os.listdir(category_path):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(category_path, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

# Datasets and loaders
train_dataset = CIFakeDataset(TRAIN_IMG_DIR, transform=transform)
test_dataset = CIFakeDataset(TEST_IMG_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# Model setup
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
scaler = torch.cuda.amp.GradScaler()

def train(model, train_loader, criterion, optimizer, scheduler, epochs=10, patience=5):
    best_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    no_improve_epochs = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model_wts)

def evaluate(model, test_loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    misclassified = []
    correct = total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = correct / total
    print(f"\n✅ Test Accuracy: {accuracy:.4f}")
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=["Fake", "Real"]))

    # Save confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(EFFNET_CM_PNG)
    plt.close()

    # ROC
    probs_real = np.array(all_probs)[:, 1]
    fpr, tpr, _ = roc_curve(all_labels, probs_real)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(EFFNET_ROC_PNG)
    plt.close()

    # Save probability predictions
    df = pd.DataFrame({
        "Image_Path": test_loader.dataset.image_paths,
        "Actual": all_labels,
        "EffNet_Prob_Fake": [p[0] for p in all_probs],
        "EffNet_Prob_Real": [p[1] for p in all_probs],
        "EffNet_Pred": all_preds
    })
    df.to_csv(EFFNET_PRED_CSV, index=False)
    print(f"Saved EfficientNet predictions to {EFFNET_PRED_CSV}")

    # UMAP & TSNE
    extract_and_plot_embeddings(model, test_loader)

def extract_and_plot_embeddings(model, loader):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)
            feats = model.forward_features(images)
            pooled = torch.nn.functional.adaptive_avg_pool2d(feats, 1)
            pooled = pooled.view(pooled.size(0), -1)
            features.append(pooled.cpu().numpy())
            labels.extend(lbls.numpy())

    features = np.concatenate(features)
    labels = np.array(labels)

    for method in ["tsne", "pca"]:
        reducer = TSNE(n_components=2, random_state=42) if method == "tsne" else PCA(n_components=2)
        reduced = reducer.fit_transform(features)
        plt.figure(figsize=(7, 5))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="coolwarm", alpha=0.6)
        plt.legend(*scatter.legend_elements(), title="Class")
        plt.title(f"{method.upper()} of EfficientNet Embeddings")
        plt.tight_layout()
        save_path = EFFNET_TSNE_PNG if method == "tsne" else EFFNET_UMAP_PNG
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    train(model, train_loader, criterion, optimizer, scheduler, epochs=10, patience=5)
    evaluate(model, test_loader)
    torch.save(model.state_dict(), EFFNET_MODEL_PATH)
    print(f"💾 Model saved to: {EFFNET_MODEL_PATH}")
