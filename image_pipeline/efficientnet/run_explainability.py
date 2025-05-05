import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from lime import lime_image
from captum.attr import IntegratedGradients
from skimage.segmentation import mark_boundaries
from efficientnet_pytorch import EfficientNet
import shap

# === Setup
model_path = "/home/bna36/misinfo_detection/models/effnet_model.pth" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_path = r"/home/bna36/misinfo_detection/data/cifake_dataset/test/fake/454 (10).jpg"
save_dir = "/home/bna36/misinfo_detection/image_pipeline/efficientnet/outputs"

os.makedirs(save_dir, exist_ok=True)

from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_name('efficientnet-b0')
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()



# Preprocess
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
original_image = Image.open(image_path).convert("RGB")
input_tensor = preprocess(original_image).unsqueeze(0).to(device)

class_names = ["Fake", "Real"]

def batch_predict(images):
    model.eval()
    batch = torch.stack([preprocess(Image.fromarray(img)).to(device) for img in images], dim=0)
    with torch.no_grad():
        logits = model(batch)
    probs = torch.nn.functional.softmax(logits, dim=1)
    return probs.cpu().numpy()

# === 1. LIME
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(
    np.array(original_image),
    batch_predict,
    top_labels=2,
    hide_color=0,
    num_samples=1000
)
temp, mask = explanation.get_image_and_mask(
    label=explanation.top_labels[0],
    positive_only=True,
    hide_rest=False,
    num_features=5,
    min_weight=0.0
)
plt.imshow(mark_boundaries(temp / 255.0, mask))
plt.axis('off')
plt.title('LIME Explanation')
plt.savefig(os.path.join(save_dir, "lime_explanation.png"))
plt.close()

# === 2. Integrated Gradients
ig = IntegratedGradients(model)

attributions, delta = ig.attribute(
    input_tensor,
    target=explanation.top_labels[0],
    baselines=torch.zeros_like(input_tensor),
    return_convergence_delta=True
)

attr = attributions.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
plt.imshow(np.clip(attr, 0, 1))
plt.axis('off')
plt.title('Integrated Gradients Attribution')
plt.savefig(os.path.join(save_dir, "integrated_gradients.png"))
plt.close()

# === 3. SHAP
background = input_tensor[:1]  # baseline
e = shap.DeepExplainer(model, background)

shap_values = e.shap_values(input_tensor)

shap.image_plot([shap_values[explanation.top_labels[0]]], np.array(original_image).reshape(1, 224, 224, 3) / 255.0)
plt.savefig(os.path.join(save_dir, "shap_explanation.png"))
plt.close()

print("✅ Explanations generated and saved successfully!")
