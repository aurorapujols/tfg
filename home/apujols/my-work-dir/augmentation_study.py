from self_supervised_training.py import *

def make_pair_transform(aug1, aug2):
    x_i = torch.stack([aug1(T.ToPILImage()(img)) for img in batch])
    x_j = torch.stack([aug2(T.ToPILImage()(img)) for img in batch])
    return x_i, x_j

if __name__ == "__main__":

    AUGMENTATIONS = {
        "rotation": T.RandomRotation(degrees=(-10, 10)),
        "blur": T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.5, 2.0))]),
        "brightness": T.RandomApply([T.ColorJitter(brightness=0.3)]),
        "contrast": T.RandomApply([T.ColorJitter(contrast=0.3)]),
        "crop": T.RandomResizedCrop(size=64, scale=(0.8, 1.0))
    }

    results = {}

    for name_i, aug_i in AUGMENTATIONS.items():
        for name_j, aug_j in AUGMENTATIONS.items():

            print(f"Training with {name_i} vs {name_j}")
            augment_fn = make_pair_transform(aug_i, aug_j)

            model = Model().to(device)
            model, _, _ = train_ssl(
                model=model,
                batch_size=64,
                num_epochs=20,
                patience=5,
                cutoff_ratio=0.001,
                lr=1e-3,
                loader=ssl_loader,
                train_loader=train_loader,
                val_loader=val_loader,
                augment_fn=augment_fn   # NEW
            )

            train_feats, train_labels = extract_backbone_features(model, val_loader, device)
            val_feats, val_labels = extract_backbone_features(model, val_loader, device)

            clf = LogisticRegression(max_iter=1000)
            clf.fit(train_feats, train_labels)
            val_pred = clf.predict(val_feats)
            acc = accuracy_score(val_labels, val_pred)

            results[(name_i, name_j)] = acc

    
    labels = list(AUGMENTATIONS.keys())
    matrix = pd.DataFrame(index=labels, columns=labels)

    for (i,j), acc in results.items():
        matrix.loc[i,j] = acc

    matrix = matrix.astype(float)

    plt.figure(figsize=(10, 8)) 
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis") 
    plt.title("Linear Probe Accuracy by Augmentation Pair") 
    plt.xlabel("Branch B") 
    plt.ylabel("Branch A") 
    plt.tight_layout() 
    plt.savefig("logs/training/augmentation_heatmap.png", dpi=300) 
    plt.close()