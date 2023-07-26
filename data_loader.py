import torch
import torchvision.transforms as trn
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
import wandb
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import glob
import numpy as np
from PIL import Image
from typing import Tuple, Any
from pathlib import Path
from typing import Optional, Callable
from torchvision.datasets import VisionDataset
from transformations import RandomResizedCrop, RandomHorizontalFlip, Compose
project_name = "TimeTuning_v2"



class VOCDataset(Dataset):

    def __init__(
            self,
            root: str,
            image_set: str = "trainaug",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            return_masks: bool = False
    ):
        super(VOCDataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
        self.image_set = image_set
        if self.image_set == "trainaug" or self.image_set == "train":
            seg_folder = "SegmentationClassAug"
        elif self.image_set == "val":
            seg_folder = "SegmentationClass"
        else:
            raise ValueError(f"No support for image set {self.image_set}")
        seg_dir = os.path.join(root, seg_folder)
        image_dir = os.path.join(root, 'images')
        if not os.path.isdir(seg_dir) or not os.path.isdir(image_dir) or not os.path.isdir(root):
            raise RuntimeError('Dataset not found or corrupted.')
        splits_dir = os.path.join(root, 'sets')
        split_f = os.path.join(splits_dir, self.image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(seg_dir, x + ".png") for x in file_names]
        self.return_masks = return_masks

        assert all([Path(f).is_file() for f in self.masks]) and all([Path(f).is_file() for f in self.images])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.images[index]).convert('RGB')
        if self.return_masks:
            mask = Image.open(self.masks[index])
        if self.image_set == "val":
            if self.transform:
                img = self.transform(img)
            if self.transforms:
                img, mask = self.transforms(img, mask)
            return img, mask
        elif "train" in self.image_set:
            if self.transform:
                img = self.transform(img)
            if self.transforms:
                res = self.transforms(img, mask)
                return res
            return img

    def __len__(self) -> int:
        return len(self.images)



class PascalVOCDataModule():
    """ 
    DataModule for Pascal VOC dataset

    Args:
        batch_size (int): batch size
        train_transform (torchvision.transforms): transform for training set
        val_transform (torchvision.transforms): transform for validation set
        test_transform (torchvision.transforms): transform for test set
        dir (str): path to dataset
        year (str): year of dataset
        split (str): split of dataset
        num_workers (int): number of workers for dataloader

    """

    def __init__(self, batch_size, train_transform, val_transform, test_transform,  dir="/ssdstore/ssalehi/VOCSegmentation", num_workers=0) -> None:
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dir = dir
        self.image_train_transform = train_transform["img"]
        self.image_val_transform = val_transform["img"]
        self.image_test_transform = test_transform["img"]
        self.target_train_transform = None
        self.target_val_transform = None
        self.target_test_transform = None
        self.shared_train_transform = train_transform["shared"]
        self.shared_val_transform = val_transform["shared"]
        self.shared_test_transform = test_transform["shared"]

    def setup(self):
        download = False
        if os.path.isdir(self.dir) == False:
            download = True
        self.train_dataset = VOCDataset(self.dir, image_set="trainaug", transform=self.image_train_transform, target_transform=self.target_train_transform, transforms=self.shared_train_transform, return_masks=True)
        self.val_dataset = VOCDataset(self.dir, image_set="val", transform=self.image_val_transform, target_transform=self.target_val_transform, transforms=self.shared_val_transform, return_masks=True)
        self.test_dataset = VOCDataset(self.dir, image_set="val", transform=self.image_test_transform, target_transform=self.target_test_transform, transforms=self.shared_test_transform, return_masks=True)

    def get_train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
    
    def get_val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    
    def get_test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    
    def get_train_dataset_size(self):
        return len(self.train_dataset)

    def get_val_dataset_size(self):
        return len(self.val_dataset)

    def get_test_dataset_size(self):
        return len(self.test_dataset)

    def get_module_name(self):
        return "PascalVOCDataModule"
    
    def get_num_classes(self):
        return 21
    


def test_pascal_data_module(logger):
    min_scale_factor = 0.5
    max_scale_factor = 2.0
    brightness_jitter_range = 0.1
    contrast_jitter_range = 0.1
    saturation_jitter_range = 0.1
    hue_jitter_range = 0.1

    brightness_jitter_probability = 0.5
    contrast_jitter_probability = 0.5
    saturation_jitter_probability = 0.5
    hue_jitter_probability = 0.5

    # Create the transformation
    image_train_transform = trn.Compose([
        trn.RandomApply([trn.ColorJitter(brightness=brightness_jitter_range)], p=brightness_jitter_probability),
        trn.RandomApply([trn.ColorJitter(contrast=contrast_jitter_range)], p=contrast_jitter_probability),
        trn.RandomApply([trn.ColorJitter(saturation=saturation_jitter_range)], p=saturation_jitter_probability),
        trn.RandomApply([trn.ColorJitter(hue=hue_jitter_range)], p=hue_jitter_probability),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])
    ])

    shared_transform = Compose([
        RandomResizedCrop(size=(448, 448), scale=(min_scale_factor, max_scale_factor)),
        RandomHorizontalFlip(probability=0.1),
    ])
        
    
    # image_train_transform = trn.Compose([trn.Resize((448, 448)), trn.ToTensor(), trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])])
    # target_train_transform = trn.Compose([trn.Resize((448, 448), interpolation=trn.InterpolationMode.NEAREST), trn.ToTensor()])
    train_transforms = {"img": image_train_transform, "target": None, "shared": shared_transform}
    dataset = PascalVOCDataModule(batch_size=4, train_transform=train_transforms, val_transform=train_transforms, test_transform=train_transforms)
    dataset.setup()
    train_dataloader = dataset.get_train_dataloader()
    val_dataloader = dataset.get_val_dataloader()
    test_dataloader = dataset.get_test_dataloader()
    print(f"Train size : {len(dataset.train_dataset)}")
    print(f"Val size : {len(dataset.val_dataset)}")
    print(f"Test size : {len(dataset.test_dataset)}")
    print(f"Train dataloader size : {len(train_dataloader)}")
    print(f"Val dataloader size : {len(val_dataloader)}")
    print(f"Test dataloader size : {len(test_dataloader)}")
    for i, (x, y) in enumerate(val_dataloader):
        print(f"Train batch {i} : {x.shape}, {y.shape}")
        ## log image
        classes = torch.unique((y * 255).long())
        print(f"Number of classes : {classes}")
        logger.log({"train_batch": [wandb.Image(x[0]), wandb.Image(y[0])]})
        if i == 10:
            break


    

if __name__ == "__main__":
    ## init wandb
    logger = wandb.init(project=project_name, group="data_loader", tags="PascalVOCDataModule", job_type="eval")
    ## test data module
    test_pascal_data_module(logger)
    ## finish wandb
    logger.finish()