
from PIL import Image

import numpy as np
import pandas as pd
from scipy import spatial

import wandb
import timm
from timm.utils import AverageMeter

from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


# Dataset and DataLoaders
class DiffusionDataset(Dataset):
    def __init__(self, df, transform, data_aug=None):
        self.df = df
        self.transform = transform
        self.length = len(self.df)
        self.aug = data_aug
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(row["filepath"])
        image = self.transform(image)
        prompt = row["prompt"]
        return image, prompt

    # Random data augmentation
    def rand_aug(self):
        pass

class DiffusionCollator():
    def __init__(self):
        self.st_model = SentenceTransformer(
            "sentence-transformer/all-MiniLM-L6-v2",
            device="cpu"
        )
    
    def __call__(self, batch):
        images, prompts = zip(*batch)
        images = torch.stack(images)
        prompt_embeddings = self.st_model.encode(
            prompts,
            show_progress_bar=False,
            convert_to_tensor=True
        )
        return images, prompt_embeddings
    
# Get dataloaders
def get_dataloaders(train_split, valid_split, input_size, batch_size):
    
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = DiffusionDataset(train_split, transform)
    valid_dataset = DiffusionDataset(valid_split, transform)
    collate_fn = DiffusionCollator()

    train_loader = DataLoader(train_dataset, 
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True,
                                collate_fn=collate_fn)

    valid_loader = DataLoader(valid_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=True,
                                collate_fn=collate_fn)

    return train_loader, valid_loader

# Cosine similarity metric
def cosine_similarity(y_trues, y_preds):
    return np.mean([
        1 - spatial.distance.cosine(y_true, y_pred)
        for y_true, y_pred in zip(y_trues, y_preds)
    ])

# training function
def train(train_split, valid_split, model_name, input_size, batch_size, num_epochs, lr):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining device: {device}")

    train_loader, valid_loader = get_dataloaders(train_split, valid_split, input_size, batch_size)
    model = timm.create_model(
        model_name=model_name,
        pretrained=True,
        num_classes=384
    )

    model.set_grad_checkpointing()
    model.to(device)

    total_iters = num_epochs * len(train_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, amsgrad=True)
    criterion = torch.nn.CosineEmbeddingLoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    best_score = -1.0

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()

        train_meter = {
            "loss": AverageMeter(),
            "cos": AverageMeter()
        }

        model.train()


if __name__ == "__main__":
    pass