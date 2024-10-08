import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class ImageTextDataset(Dataset):
    def __init__(self, img_dir, txt_labels, transform=None):
        self.img_dir = img_dir
        self.txt_labels = txt_labels
        self.transform = transform

    def __len__(self):
        return len(self.txt_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.txt_labels[idx]['image'])
        image = Image.open(img_path).convert('RGB')
        text = self.txt_labels[idx]['text']
        
        if self.transform:
            image = self.transform(image)

        return image, text

def get_dataloader(img_dir, txt_labels, batch_size=32, transform=None):
    dataset = ImageTextDataset(img_dir, txt_labels, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
