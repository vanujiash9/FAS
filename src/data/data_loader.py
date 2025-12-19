import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

def get_transforms(input_size, is_train=False):
    if is_train:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomResizedCrop(input_size, scale=(0.9, 1.0)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def build_dataloaders(img_dir, input_size, batch_size, num_workers=4):
    """
    img_dir: data/data_split (chứa folder train và test)
    """
    loaders = {}
    
    # 1. Setup Train Loader
    train_dir = os.path.join(img_dir, 'train')
    if os.path.exists(train_dir):
        train_dataset = datasets.ImageFolder(
            root=train_dir,
            transform=get_transforms(input_size, is_train=True)
        )
        
        # Cân bằng dữ liệu bằng WeightedRandomSampler
        targets = train_dataset.targets
        class_counts = np.bincount(targets)
        class_weights = 1. / class_counts
        sample_weights = class_weights[targets]
        
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(sample_weights),
            replacement=True
        )
        
        loaders['train'] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

    # 2. Setup Test Loader
    test_dir = os.path.join(img_dir, 'test')
    if os.path.exists(test_dir):
        test_dataset = datasets.ImageFolder(
            root=test_dir,
            transform=get_transforms(input_size, is_train=False)
        )
        
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
    return loaders