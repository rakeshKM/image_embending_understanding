import torch
from torchvision import transforms
from data.dataset_loader import get_dataloader
from model_loader import load_model, get_image_embeddings, get_text_embeddings
from evaluation import compute_mrr

# Configuration
IMG_DIR = "path/to/images"
TXT_LABELS = [{'image': 'img1.jpg', 'text': 'description1'}, {'image': 'img2.jpg', 'text': 'description2'}]  # Example
BATCH_SIZE = 16

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load data
dataloader = get_dataloader(IMG_DIR, TXT_LABELS, batch_size=BATCH_SIZE, transform=transform)

# Models to evaluate
models_to_evaluate = ['resnet', 'vit', 'clip', 'siglip', 'vlm']

for model_name in models_to_evaluate:
    print(f"Evaluating model: {model_name}")
    model = load_model(model_name)

    all_image_embeddings = []
    all_text_embeddings = []

    # Extract embeddings for each batch
    for images, texts in dataloader:
        image_embeddings = get_image_embeddings(model, images)
        text_embeddings = get_text_embeddings(model, texts)
        
        all_image_embeddings.append(image_embeddings)
        all_text_embeddings.append(text_embeddings)

    all_image_embeddings = torch.cat(all_image_embeddings)
    all_text_embeddings = torch.cat(all_text_embeddings)

    # Compute MRR
    mrr = compute_mrr(all_image_embeddings, all_text_embeddings)
    print(f"Model {model_name} MRR: {mrr}")
