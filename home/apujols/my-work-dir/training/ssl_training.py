import os
import time
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import ExponentialLR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from losses.contrastive_loss import ContrastiveLoss
from transformations.augment import ControlledAugment, RandomAffineMeanFill
from evaluation.linear_probe import run_linear_probe
from utils.plotting import save_plot_augmentations

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

def train_ssl(model, batch_size, num_epochs, patience, cutoff_ratio, lr, loader, train_loader, val_loader, device="cpu", version=None, output_path=None):

    print("Starting SSL training...")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    lossfn = ContrastiveLoss().to(device)

    # augmentfn = Augment() #metadata_csv="../../../data/upftfg26/apujols/processed/dataset_29884.csv")
    augmentfn = ControlledAugment([
        RandomAffineMeanFill(degrees=(-90,90), scale=(0.9, 1.1)),
        T.GaussianBlur(kernel_size=3, sigma=(0.5, 2.0)),
        T.ColorJitter(brightness=1.0, contrast=1.0),
        T.RandomResizedCrop(size=256, scale=(0.5, 1.0)),    # resize to 256x256 image crop up to 50% in the image
    ])

    avg_loss = float("inf")
    patience_count = 0
    stop_epoch = num_epochs

    epochs_df = pd.DataFrame(columns=["epoch", "schedule", "loss", "accuracy", "time"])

    for epoch in range(num_epochs):

        training_start = time.time()

        model.train()
        total_loss = 0.0

        for batch_idx, (images, fnames) in enumerate(loader):
            x_i, x_j = augmentfn(images)       # CPU augment

            x_i = x_i.to(device, non_blocking=True)
            x_j = x_j.to(device, non_blocking=True)

            # For debugging the image and its augmentations
            os.makedirs(f"{output_path}", exist_ok=True)
            if epoch==0 and batch_idx==0:
                imgs_orig = images.cpu().numpy()
                imgs_i = x_i.cpu().numpy()
                imgs_j = x_j.cpu().numpy()

                B = imgs_orig.shape[0]

                # Save as image the first the augmentations of the first image
                for b in range(B):
                    if b == 0:
                        save_plot_augmentations(img_orig=imgs_orig[b, 0], img_i=imgs_i[b, 0], img_j=imgs_j[b, 0], save_path=output_path, version=version)
    
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

        # # Train linear probe to evaluate with accuracy
        linear_probe_acc, clf = run_linear_probe(train_feats, train_labels, val_feats, val_labels)

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