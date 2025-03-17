import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pickle

class AnnotatedFloorplanDataset(Dataset):
    def __init__(self, images_dir, annotations_path, img_size=256):
        self.images_dir = images_dir
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        
        with open(annotations_path, 'rb') as f:
            self.annotations = pickle.load(f)
        
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
        self.verify_annotations()

    def verify_annotations(self):
        missing_annotations = []
        for img_file in self.image_files:
            img_id = os.path.splitext(img_file)[0]
            if img_id not in self.annotations:
                missing_annotations.append(img_id)
        
        if missing_annotations:
            raise ValueError(f"Missing annotations for images: {missing_annotations}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_file)
        
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        
        img_id = os.path.splitext(img_file)[0]
        annotations = self.annotations[img_id]
        annotations_tensor = torch.tensor(annotations, dtype=torch.float32)
        
        return img, annotations_tensor