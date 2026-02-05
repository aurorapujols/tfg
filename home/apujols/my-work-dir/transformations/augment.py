import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class RandomAffineMeanFill:
    def __init__(self, degrees=(-90, 90), translate=(0.05, 0.05), scale=(0.9, 1.1)):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale

    def __call__(self, img):
        # Convert PIL → tensor
        t = TF.to_tensor(img)  # shape (1, H, W)

        # Compute mean pixel value in 0–255 range
        mean_val = float(t.mean().item() * 255.0)

        # Sample affine params
        angle = random.uniform(*self.degrees)
        translate_px = (
            self.translate[0] * img.size[0],
            self.translate[1] * img.size[1]
        )
        scale = random.uniform(*self.scale)

        # Apply affine with mean fill
        out = TF.affine(
            img,
            angle=angle,
            translate=translate_px,
            scale=scale,
            shear=[0.0, 0.0],
            fill=int(mean_val)
        )

        return out

class ControlledAugment: 
    """ Applies each augmentation independently with probability p = 1/N. 
    Ensures each view gets at least one augmentation. 
    """ 
    def __init__(self, aug_transforms): 
        self.augs = aug_transforms # list of PIL→PIL transforms 
        self.N = len(aug_transforms) 
        self.p = 1.0 / self.N # equal probability 
        
    def apply_independent(self, img): 
        applied = [] 
        
        # apply each augmentation independently with prob p 
        for aug in self.augs: 
            if random.random() < self.p: 
                img = aug(img) 
                applied.append(aug) 
                
        # ensure at least one augmentation fires 
        if len(applied) == 0: 
            aug = random.choice(self.augs) 
            img = aug(img) 
        
        return img 
        
    def one_view(self, img_tensor): 
        img = T.ToPILImage()(img_tensor)
        
        # independent stochastic augmentations 
        img = self.apply_independent(img) 
        
        return T.ToTensor()(img) 
        
    def __call__(self, batch): 
        x_i = torch.stack([self.one_view(img) for img in batch]) 
        x_j = torch.stack([self.one_view(img) for img in batch]) 
        return x_i, x_j


# For versions 1.1, 1.2, 1.3, 1.4, 1.5, 1.6.{[1,...,16]}
class Augment:
    """
    Stochastic augmentation returning two correlated views per image.
    """
    def __init__(self):
        self.train_transform = T.Compose([
            RandomAffineMeanFill(degrees=(-90,90), scale=(0.9, 1.1)),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.5, 2.0))], p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.8, contrast=0.8)], p=0.5),
            T.RandomResizedCrop(size=64, scale=(0.8, 1.0)),

            T.ToTensor()
        ])

    def __call__(self, batch):
        # batch: tensor (B, 1, H, W)
        x_i = torch.stack([self.train_transform(T.ToPILImage()(img)) for img in batch])
        x_j = torch.stack([self.train_transform(T.ToPILImage()(img)) for img in batch])
        return x_i, x_j
