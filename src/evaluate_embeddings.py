import cv2
import torch
import numpy as np
from torchvision import transforms
from data.custom_DataLoader import CustomDataLoader
from model.get_model import GetModel
from evaluation import compute_mrr
from evaluate.evaluate2 import mean_reciprocal_rank
# Configuration
IMG_DIR = "/data/rakesh/vision_pod/public_datasets/Car_parts/images"
CSV_FILE = "/data/rakesh/vision_pod/public_datasets/Car_parts/image_labels.csv"
BATCH_SIZE = 32



# Load data from the image directory and CSV file
custom_dataloader = CustomDataLoader(IMG_DIR, CSV_FILE, batch_size=BATCH_SIZE)
dataloader = custom_dataloader.get_dataloader()
# Models to evaluate
models_to_evaluate = ['vlm']

for model_name in models_to_evaluate:
    print(f"Evaluating model: {model_name}")
    model = GetModel(model_name)

    all_image_embeddings = []
    all_text_embeddings = []
    all_relevant_indices = []

    # Extract embeddings for each batch
    for i,(images, texts, label_group) in enumerate(dataloader):  # Ignore group_label for now
        print("dataloader batch :",i,images.shape,len(texts))
        with torch.no_grad():
            image_embeddings = model.get_image_embedding(images)
            input_ids,attention_mask=model.get_text_preprocessor(texts)
            text_embeddings = model.get_text_embedding(input_ids,attention_mask)
        
        all_image_embeddings.append(image_embeddings)
        all_text_embeddings.append(text_embeddings)
        all_relevant_indices.extend(label_group)
        del images, texts, image_embeddings, text_embeddings,input_ids,attention_mask
        
    all_image_embeddings = torch.cat(all_image_embeddings)
    all_text_embeddings = torch.cat(all_text_embeddings)
    all_relevant_indices= np.array(all_relevant_indices).reshape(-1, 1)
    #all_relevant_indices = torch.cat(all_relevant_indices)
    # Compute MRR
    # mrr = compute_mrr(all_image_embeddings, all_text_embeddings)
    # print(f"Model {model_name} MRR: {mrr}")

    # Calculate MRR
    mrr_score = mean_reciprocal_rank(all_text_embeddings, all_image_embeddings,all_relevant_indices)
    print("Mean Reciprocal Rank (MRR):", mrr_score)
    print(mrr_score)