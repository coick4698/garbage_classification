import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import random
from collections import defaultdict

def get_dataloaders(data_dir, batch_size=32, val_split=0.2, image_size=224, use_small=False, small_ratio=0.2, seed=42):
    """
    데이터셋 불러오기 함수. small dataset 사용 가능 (클래스별 일부 샘플만 유지)

    Parameters:
        use_small (bool): True면 클래스별 small_ratio 비율만 샘플링
    """

    # Preprocessing based on ImageNet
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    class_names = full_dataset.classes

    # small dataset
    if use_small:
        class_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(full_dataset.samples):
            class_to_indices[label].append(idx)

        selected_indices = []
        random.seed(seed)
        for label, indices in class_to_indices.items():
            k = int(len(indices) * small_ratio)
            selected_indices.extend(random.sample(indices, k))

        full_dataset = Subset(full_dataset, selected_indices)

    # train/val split
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    if isinstance(val_dataset.dataset, Subset):
        val_dataset.dataset.dataset.transform = val_transform
    else:
        val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, class_names
