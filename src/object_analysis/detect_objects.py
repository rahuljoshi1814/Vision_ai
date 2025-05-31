import torch
from torchvision import models, transforms
from PIL import Image
import os
from typing import Dict, List

# Load MobileNetV2 pretrained model
model = models.mobilenet_v2(pretrained=True)
model.eval()

# Load ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_labels = []
try:
    import urllib.request
    with urllib.request.urlopen(LABELS_URL) as f:
        imagenet_labels = [line.strip() for line in f.readlines()]
except:
    print("⚠️ Failed to download labels from ImageNet. Using fallback class indices.")

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

def classify_images(folder_path: str) -> Dict[str, str]:
    """
    Classifies all images in a folder using MobileNetV2.

    Args:
        folder_path (str): Path to folder containing segmented object images.

    Returns:
        Dict[str, str]: Dictionary mapping image filename to predicted class.
    """
    results = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                output = model(input_tensor)
                _, pred = torch.max(output, 1)
                class_name = imagenet_labels[pred.item()] if imagenet_labels else f"class_{pred.item()}"
                results[filename] = class_name

    return results
