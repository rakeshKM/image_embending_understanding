import torch
import numpy as np
from torchvision import transforms
from dataset_loader import get_dataloader
from model_loader import load_model, get_image_embeddings, get_text_embeddings
from evaluation import compute_mrr

# Custom transform function that operates on numpy array images loaded by OpenCV
class ToTensor:
    def __call__(self, image):
        # Convert the numpy array (H, W, C) to a torch tensor (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)  # Change from (H, W, C) to (C, H, W)
        return image

# Configuration
IMG_DIR = "path/to/images"
CSV_FILE = "path/to/data.csv"
BATCH_SIZE = 16

# Preprocessing (OpenCV loads images as NumPy arrays, so we customize transformations)
transform = transforms.Compose([
    transforms.Lambda(lambda image: cv2.resize(image, (224, 224))),  # Resize using OpenCV
    ToTensor(),  # Convert NumPy array to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Load data from the image directory and CSV file
dataloader = get_dataloader(IMG_DIR, CSV_FILE, batch_size=BATCH_SIZE, transform=transform)

# Models to evaluate
models_to_evaluate = ['resnet', 'vit', 'clip', 'siglip', 'vlm']

for model_name in models_to_evaluate:
    print(f"Evaluating model: {model_name}")
    model = load_model(model_name)

    all_image_embeddings = []
    all_text_embeddings = []

    # Extract embeddings for each batch
    for images, texts, _ in dataloader:  # Ignore group_label for now
        image_embeddings = get_image_embeddings(model, images)
        text_embeddings = get_text_embeddings(model, texts)
        
        all_image_embeddings.append(image_embeddings)
        all_text_embeddings.append(text_embeddings)

    all_image_embeddings = torch.cat(all_image_embeddings)
    all_text_embeddings = torch.cat(all_text_embeddings)

    # Compute MRR
    mrr = compute_mrr(all_image_embeddings, all_text_embeddings)
    print(f"Model {model_name} MRR: {mrr}")
