import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from sklearn.model_selection import ShuffleSplit
import torchvision.transforms as T


class ImageData(Dataset):
    '''
        A PyTorch Dataset class for multi-view contrastive learning.

        It supports:
        - Strong augmentations for training
        - Softer augmentations for validation
        - Deterministic transforms for testing
        - Multi-view image generation for SimCLR-style contrastive training
        '''
    def __init__(self,
                 jitter_strength=0.2, blur=False, dataset_cfg=None, transform=None, state=None):

        '''
                Initialize the dataset, create transforms, and prepare train/val/test splits.

                Args:
                    crop_sizes (tuple): Not used directly, reserved for future extensions.
                    jitter_strength (float): Strength of color jitter.
                    blur (bool): Whether to apply Gaussian blur.
                    dataset_cfg (object): Contains dataset config including data_dir.
                    transform (callable): Optional custom transform (unused).
                    state (str): One of 'train', 'val', or 'test'.
        '''

        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg

        self.data_dir = self.dataset_cfg.data_dir
        self.views = 6
        self.train_view = self._create_transform(jitter_strength=0.4, blur= True, grayscale = True)
        self.val_view = self._create_transform(jitter_strength=0.1, blur= False, grayscale = False)


        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop((224,224)),
            #transforms.RandomResizedCrop(224, scale=(1.0, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        # Load dataset
        full_dataset = datasets.ImageFolder(root=self.data_dir)
        num_samples = len(full_dataset)

        # Ratios
        train_ratio = 0.8
        val_ratio = 0.2


        splitter = ShuffleSplit(n_splits=1, test_size=(1 - train_ratio), random_state=0)
        train_idx, val_idx = next(splitter.split(np.zeros(num_samples)))


        self.dataset_split = {
            'train': torch.utils.data.Subset(full_dataset, train_idx),
            'val': torch.utils.data.Subset(full_dataset, val_idx),
            'test': torch.utils.data.Subset(full_dataset, list(train_idx)+list(val_idx))
        }

        self.dataset = self.dataset_split[state]

    def _create_transform(self, jitter_strength=0.2, blur=True, grayscale = True):
        color_jitter = T.ColorJitter(
            0.8 * jitter_strength,
            0.8 * jitter_strength,
            0.8 * jitter_strength,
            0.2 * jitter_strength
        )

        transform = [
            T.RandomHorizontalFlip(0.5),
            T.RandomResizedCrop(224, scale=(0.4, 1.0)),
            T.RandomApply([color_jitter], p=0.4),
        ]

        if blur:
            transform.append(T.GaussianBlur(kernel_size=5))

        if grayscale:
            transform.append(T.RandomGrayscale(p=0.2))

        transform += [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ]

        return T.Compose(transform)

    def __len__(self):
        return len(self.dataset)  # Correct: returns num base images

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        if self.state == 'train':
            views = [self.train_view(img) for _ in range(self.views)]
            all_views = torch.stack(views, dim=0)  # Shape: (n_views, C, H, W)
            return all_views, label

        elif self.state == 'val':

            views = [self.val_view(img) for _ in range(self.views)]
            all_views = torch.stack(views, dim=0)  # Shape: (n_views, C, H, W)
            return all_views, label

        elif self.state == 'test':
            return self.test_transform(img), label
