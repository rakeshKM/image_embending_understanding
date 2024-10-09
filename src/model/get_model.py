
from .model import VLM, CLIPModelWrapper ,SigLIPModel

class GetModel:
    def __init__(self, model_name):
        self.model_name = model_name.lower()
        self.model = self._initialize_model()

    def _initialize_model(self):
        if self.model_name == "vlm":
            return VLM()
        elif self.model_name == "clip":
            return CLIPModelWrapper()
        elif self.model_name == "siglip":
            return SigLIPModel()
        else:
            raise ValueError(f"Model '{self.model_name}' is not supported.")

    def get_image_embedding(self, image):
        return self.model.get_image_embedding(image)

    def get_text_embedding(self, input_ids, attention_mask):
        return self.model.get_text_embedding(input_ids, attention_mask)

    def get_text_preprocessor(self, text):
        return self.model.get_text_preprocessor( text )