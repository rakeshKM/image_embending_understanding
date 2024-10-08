import torch
import torchvision.models as models
from transformers import CLIPModel, ViTModel

# Function to load models
def load_model(model_name):
    if model_name == 'resnet':
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove final FC layer to get embeddings
    elif model_name == 'vit':
        model = ViTModel.from_pretrained('google/vit-base-patch16-224')
    elif model_name == 'clip':
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    elif model_name == 'siglip':
        # Assuming SigLIP is loaded similarly to CLIP (modify as needed)
        model = CLIPModel.from_pretrained("siglip/vit-base-patch32")
    elif model_name == 'vlm':
        # Custom loading for VLM if it's a specific model
        model = load_vlm_model()  # Define how to load VLM
    else:
        raise ValueError("Invalid model name")
    
    return model

def get_image_embeddings(model, images):
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'vision_model'):
            # CLIP or SigLIP use 'vision_model' for image embeddings
            image_embeddings = model.vision_model(images).last_hidden_state.mean(dim=1)
        else:
            # ResNet and ViT models
            image_embeddings = model(images).squeeze()
    return image_embeddings

def get_text_embeddings(model, texts):
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'text_model'):
            # CLIP or SigLIP use 'text_model' for text embeddings
            text_embeddings = model.text_model(texts).last_hidden_state.mean(dim=1)
        else:
            # ViT models or others can have different text embedding strategies
            raise NotImplementedError("Text embedding extraction not implemented for this model")
    return text_embeddings
