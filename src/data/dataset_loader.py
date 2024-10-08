import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
from torchvision import transforms

class ImageTextDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        """
        img_dir: Directory with all the images.
        csv_file: Path to the CSV file containing image names, titles (text), and group labels.
        transform: Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.data = pd.read_csv(csv_file)  # Load CSV
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract the image filename, title, and group label from the CSV
        img_name = self.data.iloc[idx]['image']
        text = self.data.iloc[idx]['title']
        group_label = self.data.iloc[idx]['group_label']
        
        # Load image using OpenCV
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)  # OpenCV loads image in BGR format
        
        if image is None:
            raise ValueError(f"Image {img_path} not found or cannot be opened.")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to float32 and normalize pixel values to [0, 1] range
        image = image.astype(np.float32) / 255.0

        # Apply transformations (resize, normalize, etc.)
        if self.transform:
            image = self.transform(image)

        # Return image, text description, and group label
        return image, text, group_label

def get_dataloader(img_dir, csv_file, batch_size=32, transform=None):
    """
    img_dir: Directory with all the images.
    csv_file: Path to the CSV file containing image names, titles (text), and group labels.
    batch_size: Size of each batch of data.
    transform: Transformations to apply to the images (e.g., resizing, normalization).
    """
    dataset = ImageTextDataset(img_dir, csv_file, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
