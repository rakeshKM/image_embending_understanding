import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .custom_ImageTextDataset import ImageTextDataset

class CustomDataLoader:
    def __init__(self, image_folder,csv_file, batch_size=32, shuffle=True,transform = None):
        """
        Args:
            csv_file (str): Path to the csv file with image, title, and group_label.
            image_folder (str): Directory with all the images.
            batch_size (int): Size of the batches to be generated.
            shuffle (bool): Whether to shuffle the data at every epoch.
        """
        self.image_folder = image_folder
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.shuffle = shuffle
        # Define any necessary transforms
        self.transform = transforms.Compose([
                transforms.ToTensor(),  # Convert nd_array to Tensor ,normalize, and re-arrange into H*W*C -> C*H*W
                transforms.Resize((384, 384)),  # Resize image   (384, 384)             
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
            ])

        # Create the dataset
        self.dataset = ImageTextDataset(img_dir=self.image_folder, csv_file=self.csv_file,transform=self.transform)
        # Create the DataLoader
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle,collate_fn=ImageTextDataset.collate_fn)

    def get_dataloader(self):
        """Returns the DataLoader instance."""
        return self.dataloader

if __name__ == "__main__":  # Corrected the name check
    img_dir = "/data/rakesh/vision_pod/public_datasets/Car_parts/images"
    csv_file = "/data/rakesh/vision_pod/public_datasets/Car_parts/image_labels.csv"

    # Create an instance of CustomDataLoader
    custom_dataloader = CustomDataLoader(image_folder=img_dir, csv_file=csv_file,  batch_size=32, shuffle=True)

    # Get the DataLoader
    dataloader = custom_dataloader.get_dataloader()

    # Iterate through the DataLoader
    for images, titles, group_labels in dataloader:
        print(f"Images batch shape: {images.size()}")
        print(f"Titles: {titles}")
        print(f" Labels Group: {group_labels}")  # Removed the period at the end