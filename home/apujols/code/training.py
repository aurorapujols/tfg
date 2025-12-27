import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# BACKBONE (variable-size friendly)
# -------------------------------------------------
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        
        # Global average pooling → fixed (batch, 32)
        x = x.mean(dim=[2, 3])
        return x

# -------------------------------------------------
# PROJECTION HEAD
# -------------------------------------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=32, proj_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, proj_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x, dim=1)

# -------------------------------------------------
# FULL SSL MODEL (SimCLR-style)
# -------------------------------------------------
class Model(nn.Module):
    def __init__(self, proj_dim=64):
        super().__init__()
        self.backbone = Backbone()
        self.projection_head = ProjectionHead(in_dim=32, proj_dim=proj_dim)

    def forward(self, x):
        h = self.backbone(x)              # (batch, 32)
        z = self.projection_head(h)       # (batch, proj_dim)
        return h, z

# Example shape check (assumes torch_train_x exists and is (N,1,H,W))
# torch_train_x = ...
# model_baseline = Model().to(device)
# h, z = model_baseline(torch_train_x[:1].to(device))
# print(h.shape, z.shape)

# -------------------------------------------------
# AUGMENTATION
# -------------------------------------------------
class Augment:
    """
    Stochastic data augmentation module.
    Returns two correlated views of the same batch: x_i, x_j.
    """

    def __init__(self):
        blur = T.GaussianBlur((3, 3), (0.1, 2.0))
        self.train_transform = nn.Sequential(
            T.RandomAffine(degrees=(-50, 50), translate=(0.1, 0.1), scale=(0.5, 1.5), shear=0.2),
            T.RandomPerspective(0.4, 0.5),
            T.RandomPerspective(0.2, 0.5),
            T.RandomPerspective(0.2, 0.5),
            T.RandomApply([blur], p=0.25),
            T.RandomApply([blur], p=0.25),
        )

    def __call__(self, x):
        # x: (batch, 1, H, W) tensor
        return self.train_transform(x), self.train_transform(x)

# Example augmentation visualization (assumes torch_train_unlabeled exists)
# a = Augment()
# aug_i, aug_j = a(torch_train_unlabeled[0:100].to(device))
# i = 1
# f, axarr = plt.subplots(2, 2)
# axarr[0, 0].imshow(aug_i.cpu().detach().numpy()[i, 0], cmap="gray")
# axarr[0, 1].imshow(aug_j.cpu().detach().numpy()[i, 0], cmap="gray")
# axarr[1, 0].imshow(aug_i.cpu().detach().numpy()[i+1, 0], cmap="gray")
# axarr[1, 1].imshow(aug_j.cpu().detach().numpy()[i+1, 0], cmap="gray")
# plt.show()

# -------------------------------------------------
# CONTRASTIVE LOSS (InfoNCE / SimCLR)
# -------------------------------------------------
class ContrastiveLoss(nn.Module):
    """
    Vanilla Contrastive loss (InfoNCE) as in SimCLR.
    """

    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        # Mask to remove self-similarities
        mask = ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)
        self.register_buffer("mask", mask.float())

    def calc_similarity_batch(self, a, b):
        # a, b: (batch, dim)
        representations = torch.cat([a, b], dim=0)  # (2*batch, dim)
        return F.cosine_similarity(
            representations.unsqueeze(1),  # (2*batch, 1, dim)
            representations.unsqueeze(0),  # (1, 2*batch, dim)
            dim=2,
        )  # (2*batch, 2*batch)

    def forward(self, proj_1, proj_2):
        # proj_1, proj_2: (batch, dim)
        batch_size = proj_1.shape[0]
        assert batch_size == self.batch_size, "Batch size must match ContrastiveLoss batch_size"

        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.mask * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * self.batch_size)
        return loss

# Example loss test (assumes torch_train_x exists)
# model_test = Model().to(device)
# augment = Augment()
# X_aug_i, X_aug_j = augment(torch_train_x.to(device))
# _, z_i = model_test(X_aug_i)
# _, z_j = model_test(X_aug_j)
# loss_fn_test = ContrastiveLoss(batch_size=torch_train_x.shape[0]).to(device)
# print(loss_fn_test(z_i, z_j))

# -------------------------------------------------
# TRAINING LOOP
# -------------------------------------------------
# Assumes:
#   torch_train_unlabeled: (N, 1, H, W) tensor of unlabeled images
#   torch_train_unlabeled = ...

model = Model().to(device)
model.train()

batch_size = 512
N = torch_train_unlabeled.shape[0]
epoch_size = N // batch_size  # full batches only
num_epochs = 100
patience = 5
cutoff_ratio = 0.001

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
lossfn = ContrastiveLoss(batch_size=batch_size).to(device)
augmentfn = Augment()

loss_hist = []
improvement_hist = []
schedule_hist = []

scheduler = ExponentialLR(optimizer, gamma=0.95)
patience_count = 0
avg_loss = 1e10

for epoch in range(num_epochs):
    print(f"epoch {epoch}/{num_epochs}")
    total_loss = 0.0
    loss_change = 0.0

    for j in tqdm(range(epoch_size)):
        X = torch_train_unlabeled[j * batch_size : (j + 1) * batch_size].to(device)

        X_aug_i, X_aug_j = augmentfn(X)

        optimizer.zero_grad()

        _, z_i = model(X_aug_i)
        _, z_j = model(X_aug_j)

        loss = lossfn(z_i, z_j)
        loss.backward()
        optimizer.step()

        # Optional: check loss change after update
        with torch.no_grad():
            _, z_i_new = model(X_aug_i)
            _, z_j_new = model(X_aug_j)
            new_loss = lossfn(z_i_new, z_j_new)
            loss_change += (new_loss - loss).item()

        total_loss += loss.item()
        schedule_hist.append(scheduler.get_last_lr()[0])

    scheduler.step()

    new_avg_loss = total_loss / epoch_size
    per_loss_reduction = (avg_loss - new_avg_loss) / avg_loss
    print(f"Percentage Loss Reduction: {per_loss_reduction}")

    if per_loss_reduction < cutoff_ratio:
        patience_count += 1
        print(f"Patience counter: {patience_count}")
        if patience_count > patience:
            break
    else:
        patience_count = 0

    avg_loss = new_avg_loss
    avg_improvement = loss_change / epoch_size
    loss_hist.append(avg_loss)
    improvement_hist.append(avg_improvement)

    print(f"Average Loss: {avg_loss}")
    print(f"Average Loss change (if calculated): {avg_improvement}")

# Visualization
plt.plot(schedule_hist, label="learning rate")
plt.legend()
plt.show()

plt.plot(loss_hist, label="loss")
plt.legend()
plt.show()
