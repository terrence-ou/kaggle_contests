from PIL import Image
import random

import numpy as np
import pandas as pd
from scipy import spatial

import wandb
import timm
from timm.utils import AverageMeter
from tqdm import tqdm

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
            "sentence-transformers/all-MiniLM-L6-v2",
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
    
    transform_train = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_valid = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = DiffusionDataset(train_split, transform_train)
    valid_dataset = DiffusionDataset(valid_split, transform_valid)
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
def train(train_split, valid_split, 
          model_name, input_size, 
          batch_size, num_epochs, lr,
          use_wandb=False):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining device: {device}")

    # Getting dataloaders
    train_loader, valid_loader = get_dataloaders(train_split, valid_split, input_size, batch_size)
    
    # Setting up model
    model = timm.create_model(
        model_name=model_name,
        pretrained=True,
        num_classes=384
    )

    model.set_grad_checkpointing()
    model.to(device)

    total_iters = num_epochs * len(train_loader)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    criterion = torch.nn.CosineEmbeddingLoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters, eta_min=0)
    scaler = torch.cuda.amp.GradScaler()

    best_score = -1.0

    # Training loop
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()

        train_meters = {
            "loss": AverageMeter(),
            "cos": AverageMeter()
        }

        model.train()
        train_bar = tqdm(total=len(train_loader), dynamic_ncols=True,
                         leave=False, position=0, desc="Train")

        # A single pass train
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            
            with torch.autocast(device_type="cuda"):
                X_out = model(X)
                target = torch.ones(X.shape[0], device=device)
                loss = criterion(X_out, y, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()

            scheduler.step()

            train_loss = loss.item()
            train_cos = cosine_similarity(
                X_out.detach().cpu().numpy(),
                y.detach().cpu().numpy()
            )

            train_meters["loss"].update(train_loss, n=X.shape[0])
            train_meters["cos"].update(train_cos, n=X.shape[0])

            curr_lr = float(optimizer.param_groups[0]["lr"]) 
            train_bar.set_postfix(
                loss="{:.04f}".format(train_meters["loss"].avg),
                cos="{:.04f}".format(train_meters["cos"].avg),
                lr="{:.06f}".format(curr_lr)
            )
            train_bar.update()
        train_bar.close()

        print("Epoch {:d}\n\t train loss={:.4f}, train cos={:.4f}, lr={:.6f}".format(
            epoch + 1,
            train_meters["loss"].avg,
            train_meters["cos"].avg,
            curr_lr))

        # A single pass validation
        valid_bar = tqdm(total=len(valid_loader), dynamic_ncols=True,
                         leave=False, position=0, desc="Valid")
        
        valid_meters = {
            "loss": AverageMeter(),
            "cos": AverageMeter()
        }

        model.eval()
        for X, y in valid_loader:
            X, y = X.to(device), y.to(device)

            with torch.no_grad():
                X_out = model(X)
                target = torch.ones(X.shape[0], device=device)
                loss = criterion(X_out, y, target)

                val_loss = loss.item()
                val_cos = cosine_similarity(X_out.detach().cpu().numpy(),
                                            y.detach().cpu().numpy())
            
            valid_meters["loss"].update(val_loss, n=X.shape[0])
            valid_meters["cos"].update(val_cos, n=X.shape[0])

            valid_bar.set_postfix(
                loss="{:.04f}".format(valid_meters["loss"].avg),
                cos="{:.04f}".format(valid_meters["cos"].avg)
            )
            valid_bar.update()
        valid_bar.close()

        print("\t val loss={:.4f}, val cos={:.4f}".format(
            valid_meters["loss"].avg,
            valid_meters["cos"].avg))

        if use_wandb:
            wandb.log({"train_loss": train_meters["loss"].avg,
                       "train_cos": train_meters["cos"].avg,
                       "valid_loss": valid_meters["loss"].avg,
                       "valid_cos": valid_meters["cos"].avg,
                       "learning_rate": curr_lr})
        
        if valid_meters["cos"].avg > best_score:
            best_score = valid_meters["cos"].avg
            print("Saving model...")
            torch.save(model.state_dict(), f"{model_name}.pth")
        
        del X, y


if __name__ == "__main__":
    # Setting up WandB
    run = wandb.init(
        project="diffusion-to-prompt",
        notes="large model, SGD",
        tags=["large model", "SGD"]
    )

    wandb.config = {
        "model_name": "vit_large_patch16_224",
        "input_size": 224,
        "batch_size": 64,
        "num_epochs": 10,
        "lr": 3e-2
    }
    
    #Reading dataframe
    df = pd.read_csv("diffusiondf.csv")
    
    train_split, valid_split = train_test_split(df, test_size=0.1)
    train(train_split, valid_split, 
          wandb.config["model_name"],
          wandb.config["input_size"],
          wandb.config["batch_size"],
          wandb.config["num_epochs"],
          wandb.config["lr"],
          use_wandb=True
          )