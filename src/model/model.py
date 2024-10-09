import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, ViTModel
from transformers import CLIPProcessor, CLIPModel

class VLM(nn.Module):
    def __init__(self, vision_model_name='google/vit-large-patch32-384', text_model_name='bert-base-uncased'):
        super(VLM, self).__init__()
        
        # Vision encoder (Using a large ViT model with higher patch resolution for advanced image encoding)
        self.vision_model = ViTModel.from_pretrained(vision_model_name)
        
        # Text encoder (Using BERT for text encoding)
        self.text_model = BertModel.from_pretrained(text_model_name)
        
        # Project vision and text embeddings into a shared space
        self.vision_projection = nn.Linear(self.vision_model.config.hidden_size, 512)
        self.text_projection = nn.Linear(self.text_model.config.hidden_size, 512)
        
        # Normalization layer
        self.norm_layer = nn.LayerNorm(512)

    def get_image_embedding(self, images):
        """
        Forward pass through the vision encoder.
        Args:
            images: Input images as tensors (B, C, H, W)
        Returns:
            Projected image embeddings (B, 512)
        """
        # Pass the images through the vision transformer
        vision_outputs = self.vision_model(pixel_values=images)
        vision_embeddings = vision_outputs.last_hidden_state.mean(dim=1)  # Global average pooling

        # Project the embeddings to a shared latent space
        projected_vision_embeddings = self.vision_projection(vision_embeddings)
        normalized_vision_embeddings = self.norm_layer(projected_vision_embeddings)
        
        return normalized_vision_embeddings

    def get_text_embedding(self, input_ids, attention_mask):
        """
        Forward pass through the text encoder.
        Args:
            input_ids: Input token IDs for the text (B, L)
            attention_mask: Attention mask for the input text (B, L)
        Returns:
            Projected text embeddings (B, 512)
        """
        # Pass the input text through BERT
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = text_outputs.last_hidden_state.mean(dim=1)  # Global average pooling

        # Project the embeddings to a shared latent space
        projected_text_embeddings = self.text_projection(text_embeddings)
        normalized_text_embeddings = self.norm_layer(projected_text_embeddings)
        
        return normalized_text_embeddings
    def get_text_preprocessor(self,texts):

        # Prepare the tokenizer for text
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Sample text descriptions,a list
        # texts = ["A picture of a cat", "A car on the road", "A group of people", "A beautiful landscape", 
        #         "A sunny beach", "A dog running", "A mountain range", "A person riding a bicycle"]

        # Tokenize the text inputs
        text_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        input_ids = text_inputs['input_ids']
        attention_mask = text_inputs['attention_mask']
        return input_ids,attention_mask

    def compute_similarity(self, image_embeddings, text_embeddings):
        """
        Compute cosine similarity between image and text embeddings.
        Args:
            image_embeddings: Image embeddings from the vision model (B, 512)
            text_embeddings: Text embeddings from the text model (B, 512)
        Returns:
            Cosine similarity scores (B, B)
        """
        image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        return torch.matmul(image_embeddings, text_embeddings.T)

    def forward(self, images, input_ids, attention_mask):
        """
        Full forward pass for the VLM model. This computes both image and text embeddings.
        Args:
            images: Input images (B, C, H, W)
            input_ids: Token IDs for the text (B, L)
            attention_mask: Attention mask for the input text (B, L)
        Returns:
            Cosine similarity scores (B, B)
        """
        # Get image and text embeddings
        image_embeddings = self.get_image_embedding(images)
        text_embeddings = self.get_text_embedding(input_ids, attention_mask)
        
        # Compute cosine similarity between images and text embeddings
        similarity = self.compute_similarity(image_embeddings, text_embeddings)
        return image_embeddings,text_embeddings,similarity


class CLIPModelWrapper(nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch16'):
        super(CLIPModelWrapper, self).__init__()
        
        # Load the pre-trained CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
    def get_image_embedding(self, images):
        """
        Forward pass through the CLIP vision encoder.
        Args:
            images: Input images as tensors (B, C, H, W)
        Returns:
            Image embeddings (B, 512)
        """
        image_embeddings = self.clip_model.get_image_features(pixel_values=images)
        normalized_image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        return normalized_image_embeddings

    def get_text_embedding(self, input_ids, attention_mask):
        """
        Forward pass through the CLIP text encoder.
        Args:
            input_ids: Input token IDs for the text (B, L)
            attention_mask: Attention mask for the input text (B, L)
        Returns:
            Text embeddings (B, 512)
        """
        text_embeddings = self.clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        normalized_text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        return normalized_text_embeddings

    def get_text_preprocessor(self,texts):
        # Prepare the processor for text input
        processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

        # Sample text descriptions
        # texts = ["A picture of a cat", "A car on the road", "A group of people", "A beautiful landscape", 
        #         "A sunny beach", "A dog running", "A mountain range", "A person riding a bicycle"]

        # Tokenize the text inputs
        text_inputs = processor(text=texts, padding=True, truncation=True, return_tensors='pt')
        input_ids = text_inputs['input_ids']
        attention_mask = text_inputs['attention_mask']
        return input_ids,attention_mask

    def compute_similarity(self, image_embeddings, text_embeddings):
        """
        Compute cosine similarity between image and text embeddings.
        Args:
            image_embeddings: Image embeddings from the vision model (B, 512)
            text_embeddings: Text embeddings from the text model (B, 512)
        Returns:
            Cosine similarity scores (B, B)
        """
        return torch.matmul(image_embeddings, text_embeddings.T)

    def forward(self, images, input_ids, attention_mask):
        """
        Full forward pass for the CLIP model. This computes both image and text embeddings.
        Args:
            images: Input images (B, C, H, W)
            input_ids: Token IDs for the text (B, L)
            attention_mask: Attention mask for the input text (B, L)
        Returns:
            Cosine similarity scores (B, B)
        """
        # Get image and text embeddings
        image_embeddings = self.get_image_embedding(images)
        text_embeddings = self.get_text_embedding(input_ids, attention_mask)
        
        # Compute cosine similarity between images and text embeddings
        similarity = self.compute_similarity(image_embeddings, text_embeddings)
        return image_embeddings,text_embeddings,similarity


class SigLIPModel(nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        super(SigLIPModel, self).__init__()
        
        # Load the pre-trained CLIP model and processor (we'll modify projection)
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Add custom projection layers to refine embeddings
        self.image_projection = nn.Linear(self.clip_model.config.vision_config.hidden_size, 512)
        self.text_projection = nn.Linear(self.clip_model.config.text_config.hidden_size, 512)
        
        # Extra normalization layer (SigLIP's custom normalization)
        self.norm_layer = nn.LayerNorm(512)

    def get_image_embedding(self, images):
        """
        Forward pass through the SigLIP vision encoder (modified CLIP).
        Args:
            images: Input images as tensors (B, C, H, W)
        Returns:
            Image embeddings (B, 512)
        """
        image_embeddings = self.clip_model.get_image_features(pixel_values=images)
        
        # Project the embeddings to a shared latent space
        projected_image_embeddings = self.image_projection(image_embeddings)
        normalized_image_embeddings = self.norm_layer(projected_image_embeddings)
        
        return normalized_image_embeddings

    def get_text_embedding(self, input_ids, attention_mask):
        """
        Forward pass through the SigLIP text encoder (modified CLIP).
        Args:
            input_ids: Input token IDs for the text (B, L)
            attention_mask: Attention mask for the input text (B, L)
        Returns:
            Text embeddings (B, 512)
        """
        text_embeddings = self.clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        
        # Project the embeddings to a shared latent space
        projected_text_embeddings = self.text_projection(text_embeddings)
        normalized_text_embeddings = self.norm_layer(projected_text_embeddings)
        
        return normalized_text_embeddings

    def get_text_preprocessor(self,texts):
        # Prepare the processor for text input
        processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

        # Sample text descriptions
        # texts = ["A picture of a cat", "A car on the road", "A group of people", "A beautiful landscape", 
        #         "A sunny beach", "A dog running", "A mountain range", "A person riding a bicycle"]

        # Tokenize the text inputs
        text_inputs = processor(text=texts, padding=True, truncation=True, return_tensors='pt')
        input_ids = text_inputs['input_ids']
        attention_mask = text_inputs['attention_mask']
        return input_ids,attention_mask

    def compute_similarity(self, image_embeddings, text_embeddings):
        """
        Compute cosine similarity between image and text embeddings.
        Args:
            image_embeddings: Image embeddings from the vision model (B, 512)
            text_embeddings: Text embeddings from the text model (B, 512)
        Returns:
            Cosine similarity scores (B, B)
        """
        return torch.matmul(image_embeddings, text_embeddings.T)

    def forward(self, images, input_ids, attention_mask):
        """
        Full forward pass for the SigLIP model. This computes both image and text embeddings.
        Args:
            images: Input images (B, C, H, W)
            input_ids: Token IDs for the text (B, L)
            attention_mask: Attention mask for the input text (B, L)
        Returns:
            Cosine similarity scores (B, B)
        """
        # Get image and text embeddings
        image_embeddings = self.get_image_embedding(images)
        text_embeddings = self.get_text_embedding(input_ids, attention_mask)
        
        # Compute cosine similarity between images and text embeddings
        similarity = self.compute_similarity(image_embeddings, text_embeddings)
        return image_embeddings,text_embeddings,similarity

if __name__ == "__main__":


    # # Initialize the VLM model
    # vlm_model = VLM()

    # # Prepare the tokenizer for text
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # # Sample images (you can load real images here)
    # images = torch.randn(8, 3, 384, 384)  # 8 sample images with size 384x384 (C, H, W)

    # # Sample text descriptions
    # texts = ["A picture of a cat", "A car on the road", "A group of people", "A beautiful landscape", 
    #         "A sunny beach", "A dog running", "A mountain range", "A person riding a bicycle"]

    # # Tokenize the text inputs
    # text_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    # input_ids = text_inputs['input_ids']
    # attention_mask = text_inputs['attention_mask']

    # # Forward pass through the VLM model
    # mage_embeddings,text_embeddings,similarity_scores = vlm_model(images, input_ids, attention_mask)

    # print(similarity_scores)


    # Initialize CLIP and SigLIP models
    clip_model = CLIPModelWrapper()
    siglip_model = SigLIPModel()

    # Prepare the processor for text input
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    # Sample images (replace with actual images)
    images = torch.randn(8, 3, 224, 224)  # Sample images (B, C, H, W)

    # Sample text descriptions
    texts = ["A picture of a cat", "A car on the road", "A group of people", "A beautiful landscape", 
            "A sunny beach", "A dog running", "A mountain range", "A person riding a bicycle"]

    # Tokenize the text inputs
    text_inputs = processor(text=texts, padding=True, truncation=True, return_tensors='pt')
    input_ids = text_inputs['input_ids']
    attention_mask = text_inputs['attention_mask']

    # Forward pass through CLIP and SigLIP models
    #img_emb,text_emb,clip_similarity_scores = clip_model(images, input_ids, attention_mask)
    img_emb,text_emb,siglip_similarity_scores = siglip_model(images, input_ids, attention_mask)

    # Output similarity scores
    print("CLIP Similarity Scores:")
    #print(clip_similarity_scores)

    print("SigLIP Similarity Scores:")
    #print(siglip_similarity_scores)