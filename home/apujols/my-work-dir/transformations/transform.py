import torchvision.transforms as T

base_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])