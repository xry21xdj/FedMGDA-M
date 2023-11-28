# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 20:53:14 2023

@author: 56516
"""
import pickle
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import Dataset
from PIL import Image

class SubImageNet(Dataset):
    """
    Constructs a subset of ImageNet dataset from a pickle file;
    expects pickle file to store list of indices.

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """
    def __init__(self, path, imagenet_data=None, imagenet_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices.
        :param imagenet_data: ImageNet dataset inputs.
        :param imagenet_targets: ImageNet dataset labels.
        :param transform:
        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        if transform is None:
            self.transform = \
                Compose([
                    ToTensor(),
                    # Normalize with ImageNet mean and std values
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

        if imagenet_data is None or imagenet_targets is None:
            self.data, self.targets = self.get_imagenet()  # TODO: Implement this method to get ImageNet data
        else:
            self.data, self.targets = imagenet_data, imagenet_targets

        # Note: For ImageFolder dataset, the data and targets might be stored differently
        # The below lines might need adjustment based on how you structure the get_imagenet function.
        self.data = [self.data[i] for i in self.indices]
        self.targets = [self.targets[i] for i in self.indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # If not already a PIL image (depends on your get_imagenet implementation)
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index

    from torchvision.datasets import ImageFolder
    
    def get_imagenet(self):
        """
        Loads the ImageNet dataset using torchvision's ImageFolder.
    
        :return: A tuple of (data, targets), where data is a list of image paths
                 and targets is a list of class indices.
        """
        # Define the path to your ImageNet (or Tiny ImageNet) dataset
        dataset_path = "/path/to/your/imagenet/directory"
        
        # Load the dataset
        imagenet_dataset = ImageFolder(dataset_path)
        
        # In the ImageFolder dataset, the data is a tuple of (image, class_index).
        # Therefore, we split the data and targets.
        data = [sample[0] for sample in imagenet_dataset]
        targets = [sample[1] for sample in imagenet_dataset]
    
        return data, targets

