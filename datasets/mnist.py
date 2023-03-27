import os

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class ImagePairDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('_blur.png')]
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_name = image_file.split('_blur.png')[0]
        X = Image.open(os.path.join(self.root_dir, f'{image_name}.png'))
        X_blur = Image.open(os.path.join(self.root_dir, image_file))
        X = self.transform(X)
        X_blur = self.transform(X_blur)
        return X, X_blur


def load_mnist(train_path, test_path):
    train_dataset = ImagePairDataset(train_path)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = ImagePairDataset(test_path)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

    return train_loader, test_loader
