import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from torch.optim.lr_scheduler import ExponentialLR
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset


# version = "1.1"     # SSL   Original Crop   Out-dim:    64,  128, 256
                      #                       Scale-dim:  32,  64,  64
                      #                       Proj-dim:   128, 256, 512
# version = "1.2"     # SSL   global_threshold      Out-dim: 256, Scale-dim: 64, Proj-dim: 512
# version = "1.3"     # SSL   meteors_stretch       Out-dim: 256, Scale-dim: 64, Proj-dim: 512
# version = "1.4"     # SSL   min_max_stretch       Out-dim: 256, Scale-dim: 64, Proj-dim: 512
# version = "1.5"     # SSL   percentile_stretch    Out-dim: 256, Scale-dim: 64, Proj-dim: 512

parser = argparse.ArgumentParser()
parser.add_argument("--scale_dim", type=int, default=64)
parser.add_argument("--backbone_dim", type=int, default=256)
parser.add_argument("--projection_dim", type=int, default=512)
parser.add_argument("--version", type=str, default="1.1")
args = parser.parse_args()

SCALE_DIM = args.scale_dim
BACKBONE_DIM = args.backbone_dim
PROJECTION_DIM = args.projection_dim
VERSION = args.version

# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# DATASET (variable-size friendly)
# -------------------------------------------------
class MyMeteorDatasetLabeled(Dataset):
    def __init__(self, image_dir, csv_file=None, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        df = pd.read_csv(csv_file, sep=";")

        # Keep only files that exist
        ending = "_CROP_SUMIMG" if VERSION == "1.1" else "_CROP_ENHANCED"
        valid_files = set(os.path.splitext(f)[0].replace(ending, "") for f in os.listdir(image_dir) if f.lower().endswith(".png"))   
        df = df[df["filename"].isin(valid_files)]

        self.df = df.sort_values("filename").reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ending = "_CROP_SUMIMG" if VERSION == "1.1" else "_CROP_ENHANCED"
        img_path = os.path.join(self.image_dir, row['filename'] + f"{ending}.png")    
        img = Image.open(img_path).convert('L')
        label = row['class']

        if self.transform:
            img = self.transform(img)

        return img, label

class MyMeteorDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform

        # Sort filenames
        ending = "_CROP_SUMIMG" if VERSION == "1.1" else "_CROP_ENHANCED"
        self.files = sorted([os.path.splitext(f)[0].replace(ending, "") for f in os.listdir(folder) if f.lower().endswith(('.png'))])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        ending = "_CROP_SUMIMG" if VERSION == "1.1" else "_CROP_ENHANCED"
        img_path = os.path.join(self.folder, fname + f"{ending}.png") 
        img = Image.open(img_path).convert('L')

        if self.transform:
            img = self.transform(img)

        return img, fname     # return filename as label

# Needed because the images are of different sizes:
def pad_collate(batch):
    """
    batch: list of (img, filename)
    img shape: (1, H, W)
    """
    images, filenames = zip(*batch) # ((img1, fname1), (img2, fname2)) -> ((img1, img2), (fname1, fname2))

    # Find max height and width in this batch
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    # Make each image as big as the max width and height in the batch adding black pixels in the right and bottom
    padded_images = []
    for img in images:
        _, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w

        # Pad: (left, right, top, bottom)
        padded = F.pad(img, (0, pad_w, 0, pad_h))
        padded_images.append(padded)

    # stack images of size (C, max_h, max_w) into a tensor of size (batch_size, C, max_h, max_w)
    return torch.stack(padded_images), filenames

# -------------------------------------------------
# BACKBONE (variable-size friendly)
# -------------------------------------------------
class Backbone(nn.Module):
    def __init__(self, out_dim=BACKBONE_DIM, scale_dim=SCALE_DIM):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, scale_dim, 3, padding=1),
            nn.BatchNorm2d(scale_dim),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(scale_dim, scale_dim*2, 3, padding=1),
            nn.BatchNorm2d(scale_dim*2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(scale_dim*2, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(self.conv2(x), 2)
        x = F.max_pool2d(self.conv3(x), 2)

        # Global average pooling → fixed (batch, out_dim)
        x = x.mean(dim=[2, 3])
        return x

# -------------------------------------------------
# PROJECTION HEAD
# -------------------------------------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=BACKBONE_DIM, hidden_dim=BACKBONE_DIM*2, proj_dim=BACKBONE_DIM*2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, proj_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.normalize(x, dim=1)

# -------------------------------------------------
# FULL SSL MODEL (SimCLR-style)
# -------------------------------------------------
class Model(nn.Module):
    def __init__(self, proj_dim=BACKBONE_DIM*2):
        super().__init__()
        self.backbone = Backbone()
        self.projection_head = ProjectionHead(in_dim=BACKBONE_DIM, proj_dim=proj_dim)

    def forward(self, x):
        h = self.backbone(x)              # (batch, backbone_dim)
        z = self.projection_head(h)       # (batch, proj_dim)
        return h, z

# -------------------------------------------------
# AUGMENTATION
# -------------------------------------------------
class Augment:
    """
    Stochastic augmentation returning two correlated views per image.
    """
    def __init__(self):
        self.train_transform = T.Compose([
            # Mild geometric transforms
            T.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.05), scale=(0.9, 1.1)),

            # Meteor-like distortions
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.5, 2.0))], p=0.3),
            T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5),

            # Random crop
            T.RandomResizedCrop(size=64, scale=(0.8, 1.0)),

            T.ToTensor()
        ])

    def __call__(self, batch):
        # batch: tensor (B, 1, H, W)
        x_i = torch.stack([self.train_transform(T.ToPILImage()(img)) for img in batch])
        x_j = torch.stack([self.train_transform(T.ToPILImage()(img)) for img in batch])
        return x_i, x_j


# -------------------------------------------------
# CONTRASTIVE LOSS (SimCLR)
# -------------------------------------------------
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss that brings embedding of positive paris together. Uses dynamic batch size to
    handle different size of last batch.
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)

        z = torch.cat([z_i, z_j], dim=0)
        z = F.normalize(z, dim=1)

        sim = torch.matmul(z, z.T) / self.temperature

        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim.masked_fill_(mask, -9e15)

        positives = torch.cat([
            torch.diag(sim, batch_size),
            torch.diag(sim, -batch_size)
        ])

        denominator = sim.exp().sum(dim=1)
        loss = -torch.log(positives.exp() / denominator)

        return loss.mean()

def extract_backbone_features(model, dataloader, device):
    model.eval()
    feats, labels = [], []

    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            h, _ = model(imgs)
            feats.append(h.cpu())
            labels.extend(lbls)

    feats = torch.cat(feats, dim=0).numpy()
    labels = np.array(labels)
    return feats, labels

def train_ssl(model, batch_size, num_epochs, patience, cutoff_ratio, lr, loader, train_loader, val_loader):

    print("Starting training...")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    lossfn = ContrastiveLoss().to(device)
    augmentfn = Augment()

    avg_loss = float("inf")
    patience_count = 0
    stop_epoch = num_epochs

    epochs_df = pd.DataFrame(columns=["epoch", "schedule", "loss", "accuracy", "time"])

    for epoch in range(num_epochs):

        training_start = time.time()

        model.train()
        total_loss = 0.0

        for images, _ in loader:
            # images: padded CPU tensor (B,1,H,W)
            x_i, x_j = augmentfn(images)       # CPU augment
            x_i = x_i.to(device, non_blocking=True)
            x_j = x_j.to(device, non_blocking=True)

            optimizer.zero_grad()
            _, z_i = model(x_i)
            _, z_j = model(x_j)

            loss = lossfn(z_i, z_j)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        epoch_loss = total_loss / len(loader)
        improvement = (avg_loss - epoch_loss) / max(avg_loss, 1e-8)
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - training_start

        # Extract backbone features
        train_feats, train_labels = extract_backbone_features(model, train_loader, device)
        val_feats, val_labels = extract_backbone_features(model, val_loader, device)

        # # Train linear probe
        clf = LogisticRegression(max_iter=1000)
        clf.fit(train_feats, train_labels)

        # # Evaluate
        val_pred = clf.predict(val_feats)
        linear_probe_acc = accuracy_score(val_labels, val_pred)

        # Append to DataFrame
        epochs_df.loc[len(epochs_df)] = {
            "epoch": epoch + 1,
            "schedule": current_lr,
            "loss": epoch_loss,
            "accuracy": linear_probe_acc,
            "time": epoch_time }

        print(f"Epoch {epoch:03d} | "f"Loss: {epoch_loss:.4f} | "f"LR: {current_lr:.6e} | "f"Δ: {improvement:.6f} | "f"Accuracy: {linear_probe_acc}")

        if improvement < cutoff_ratio:
            patience_count += 1
            if patience_count >= patience:
                print("Early stopping triggered.")
                stop_epoch = epoch + 1
                break
        else:
            patience_count = 0

        avg_loss = epoch_loss

    print("SSL training complete.\n")

    return model, epochs_df, stop_epoch


if __name__ == "__main__":

    print("Device: ", device)
    print(f"BACKBONE_DIM={BACKBONE_DIM}  |  SCALE_DIM={SCALE_DIM}  |  PROJECTION_DIM={PROJECTION_DIM}   | VERSION={VERSION}")

    start = time.time()

    # --------------------------------------------
    # Input variables to change each version
    # --------------------------------------------

    input_folder = "percentile_stretch"


    # ---------------------------------------------
    # Load labeled and unlabeled datasets
    # ---------------------------------------------

    # UNLABELED DATASET
    unlabeled_dataset = MyMeteorDataset(folder=f"../../../data/upftfg26/apujols/processed/{input_folder}", transform=transforms.ToTensor()) # v1.3
    ssl_loader = DataLoader(unlabeled_dataset, batch_size=64, shuffle=True, collate_fn=pad_collate, pin_memory=True)
    print(f"Unlabeled dataset size: {len(unlabeled_dataset)}")
    imgs_u, fnames_u = next(iter(ssl_loader))

    print(f"Sample (random from 1st batch):\tFilename: {fnames_u[0]}\tImage tensor shape: {imgs_u[0].shape}")

    # LABELED DATASET
    labeled_dataset = MyMeteorDatasetLabeled(
        image_dir=f"../../../data/upftfg26/apujols/processed/{input_folder}",
        csv_file="../../../data/upftfg26/apujols/processed/dataset_temp.csv",
        transform=transforms.ToTensor())
    print("Labeled Dataset size: ", len(labeled_dataset))

    # Split into training and val split
    indices = list(range(len(labeled_dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=labeled_dataset.df["class"])

    train_set = Subset(labeled_dataset, train_idx)
    val_set = Subset(labeled_dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=True, collate_fn=pad_collate)

    imgs_l, labels_l = next(iter(train_loader))
    print(f"Sample (random from 1st batch):\tLabel: {labels_l[0]}\tImage tensor shape: {imgs_l[0].shape}")


    # ---------------------------------------------
    # Initialize and train the SSL model
    # ---------------------------------------------

    model = Model().to(device)

    # Training loop
    training_start = time.time()
    model, epochs_df, stop_epoch = train_ssl(
        model=model,
        batch_size=64,
        num_epochs=100,
        patience=5,
        cutoff_ratio=0.001,
        lr=1e-3,
        loader=ssl_loader,
        train_loader=train_loader,
        val_loader=val_loader
        )


    # -------------------------------------------------
    # Get the results of the final model
    # -------------------------------------------------
    model.eval()
    features, filenames = extract_backbone_features(model, ssl_loader, device)
    np.save(f"logs/training/ssl_features_v{VERSION}_{BACKBONE_DIM}.npy", features)
    np.save(f"logs/training/ssl_filenames_v{VERSION}_{BACKBONE_DIM}.npy", np.array(filenames))

    end = time.time()

    epochs_df.to_csv(f"logs/training/epochs_model_v{VERSION}_{BACKBONE_DIM}.csv", sep=";", index=False)

    final_acc = epochs_df["accuracy"].iloc[-1]
    final_loss = epochs_df["loss"].iloc[-1]
    training_time = epochs_df["time"].sum()
    total_time = end - start

    print("\n" + "="*60)
    print(f" SSL Experiment Summary — Version {VERSION} ")
    print("="*60)
    print(f"Input folder: {input_folder}")
    print(f"Backbone dim: {BACKBONE_DIM}")
    print(f"Stop epoch: {stop_epoch}")
    print(f"Final accuracy: {final_acc:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Total training time: {training_time:.2f} sec ({(training_time/total_time)*100}%)")
    print(f"Total time: {total_time:.2f} sec")
