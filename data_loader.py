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

project_name = "TimeTuning_v2"



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

    def __init__(self, batch_size, train_transform, val_transform, test_transform,  dir="/home/ssalehi/dataset/PascalVOC_v2", year="2012", split="train", num_workers=2) -> None:
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dir = dir
        self.split = split
        self.image_train_transform = train_transform["img"]
        self.image_val_transform = val_transform["img"]
        self.image_test_transform = test_transform["img"]
        self.target_train_transform = train_transform["target"]
        self.target_val_transform = val_transform["target"]
        self.target_test_transform = test_transform["target"]
    def setup(self):
        download = False
        if os.path.isdir(self.dir) == False:
            download = True
        self.train_dataset = VOCSegmentation(self.dir, year='2012', image_set='train', download=download, transform=self.image_train_transform, target_transform=self.target_train_transform)
        self.val_dataset = VOCSegmentation(self.dir, year='2012', image_set='val', download=download, transform=self.image_val_transform, target_transform=self.target_val_transform)
        self.test_dataset = VOCSegmentation(self.dir, year='2012', image_set='val', download=download, transform=self.image_test_transform, target_transform=self.target_test_transform)

    def get_train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def get_val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def get_test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def get_train_dataset_size(self):
        return len(self.train_dataset)

    def get_val_dataset_size(self):
        return len(self.val_dataset)

    def get_test_dataset_size(self):
        return len(self.test_dataset)

    def get_module_name(self):
        return "PascalVOCDataModule"
    


def test_pascal_data_module(logger):

    image_train_transform = trn.Compose([trn.Resize(224, 224), trn.ToTensor(), trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])])
    target_train_transform = trn.Compose([trn.Resize((224, 224), interpolation=trn.InterpolationMode.NEAREST), trn.ToTensor()])
    train_transforms = {"img": image_train_transform, "target": target_train_transform}
    dataset = PascalVOCDataModule(batch_size=4, train_transform=train_transforms, val_transform=train_transforms, test_transform=train_transforms)
    dataset.setup()
    train_dataloader = dataset.get_train_dataloader()
    val_dataloader = dataset.get_val_dataloader()
    test_dataloader = dataset.get_test_dataloader()
    print(f"Dataset size : {len(dataset.train_dataset)}")
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