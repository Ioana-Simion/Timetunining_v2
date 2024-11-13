from collections import OrderedDict
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
from image_transformations import RandomResizedCrop, RandomHorizontalFlip, Compose
import random
import json
from enum import Enum
from my_utils import denormalize_video, make_seg_maps, visualize_sampled_videos
from torch.utils.data.distributed import DistributedSampler as DistributedSampler

import video_transformations


import torch
import torchvision as tv
from abc import ABC, abstractmethod
import os
import scipy.io as sio
from torchvision.datasets import CIFAR10
from pycocotools.coco import COCO
from typing import Any, Callable, List, Optional, Tuple

from zipfile import ZipFile

import gzip

project_name = "TimeTuning_v2"





torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
from torchvision import transforms





class Dataset(torch.nn.Module):
    @abstractmethod
    def get_train_loader(self):
        pass

    @abstractmethod
    def get_test_loader(self):
        pass

    @abstractmethod
    def get_val_loader(self):
        pass

    @abstractmethod
    def get_train_dataset(self):
        pass

    @abstractmethod
    def get_test_dataset(self):
        pass

    @abstractmethod
    def get_val_dataset(self):
        pass
    
    @abstractmethod
    def get_num_classes(self):
        pass

    @abstractmethod
    def get_dataset_name(self):
        pass


class NormalSubsetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, subset_indices):
        self.dataset = dataset
        self.subset_indices = subset_indices

    def __getitem__(self, index):
        return self.dataset[self.subset_indices[index]]

    def __len__(self):
        return len(self.subset_indices)


class Cifar10_Handler(Dataset):
    def __init__(self, batch_size, normal_classes, transformations, val_transformations, num_workers, device=None):
        self.batch_size = batch_size
        self.normal_classes = normal_classes
        self.num_workers = num_workers
        self.device = device
        self.dataset_name = "Cifar10"
        self.num_classes = 10
        self.transform = transformations
        self.val_transform = val_transformations
        self.train_dataset = CIFAR10(root="/ssdstore/ssalehi/cifar", train=True, download=True, transform=self.transform)
        self.test_dataset = CIFAR10(root="/ssdstore/ssalehi/cifar", train=False, download=True, transform=self.val_transform )
        self.binarize_test_labels()
        ## split the dataset to train and validation
        normal_subset_indices, normal_subset = self.get_normal_sebset_indices()
        self.normal_dataset = NormalSubsetDataset(self.train_dataset, normal_subset_indices)
        print("Normal Subset Size: ", len(self.normal_dataset))
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.normal_dataset, [int(len(self.normal_dataset)*0.8), int(len(self.normal_dataset)*0.2)])
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader

    def get_val_loader(self):
        return self.val_loader

    def get_train_dataset(self):
        return self.train_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_val_dataset(self):
        return self.val_dataset

    def get_num_classes(self):
        return self.num_classes

    def get_dataset_name(self):
        return self.dataset_name
    
    def get_normal_classes(self):
        return self.normal_classes

    def get_normal_sebset_indices(self):
        normal_subset_indices = [i for i, (data, label) in enumerate(self.train_dataset) if label in self.normal_classes]
        normal_subset = torch.utils.data.Subset(self.train_dataset, normal_subset_indices)
        return normal_subset_indices, normal_subset
    
    def binarize_test_labels(self):
        for i, label in enumerate(self.test_dataset.targets):
            if label in self.normal_classes:
                self.test_dataset.targets[i] = 0
            else:
                self.test_dataset.targets[i] = 1
   
    



class ImangeNet_100_Handler(Dataset):
    def __init__(self, batch_size, dataset_path, transformations, val_transformations, num_workers, device=None):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.num_workers = num_workers
        self.device = device
        self.dataset_name = "ImageNet_100"
        self.num_classes = 100
        self.transform = transformations
        self.val_transform = val_transformations
        self.image_train_transform = transformations["img"]
        self.image_val_transform = transformations["img"]
        self.image_test_transform = transformations["img"]
        self.target_train_transform = None
        self.target_val_transform = None
        self.target_test_transform = None
        self.shared_train_transform = transformations["shared"]
        self.shared_val_transform = val_transformations["shared"]
        self.shared_test_transform = val_transformations["shared"]


    def setup(self):
        self.train_dataset = ImageFolder(root=f"{self.dataset_path}/train", transform=self.image_train_transform)
        self.test_dataset = ImageFolder(root=f"{self.dataset_path}/val", transform=self.val_transform )
        ## split the dataset to train and validation
        print("Normal Subset Size: ", len(self.train_dataset))
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.train_dataset, [int(len(self.train_dataset)*0.9), int(len(self.train_dataset)*0.1)])
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def get_train_dataloader(self):
        return self.train_loader

    def get_test_dataloader(self):
        return self.test_loader

    def get_val_dataloader(self):
        return self.val_loader

    def get_train_dataset(self):
        return self.train_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_val_dataset(self):
        return self.val_dataset

    def get_num_classes(self):
        return self.num_classes

    def get_dataset_name(self):
        return self.dataset_name
    
    def get_normal_classes(self):
        return self.normal_classes



class SamplingMode(Enum):
    UNIFORM = 0
    DENSE = 1
    Full = 2
    Regular = 3


def get_file_path(classes_directory):
    ## find all the folders and add all the files in the folders to a dict with keys are name of the file and values are the path to the file

    folder_file_path = {} ## key is the directory_path and value are the files in the directory
    for root, dirs, files in os.walk(classes_directory):
        for file in sorted(files):
            if file.endswith(".jpg") or file.endswith(".png"):
                if root not in folder_file_path:
                    folder_file_path[root] = []
                folder_file_path[root].append(file)
    
    return dict(sorted(folder_file_path.items()))



def make_categories_dict(meta_dict, name):
    category_list = []
    if "ytvos" in name:
        video_name_list = meta_dict["videos"].keys()
        for name in video_name_list:
            obj_list = meta_dict["videos"][name]["objects"].keys()
            for obj in obj_list:
                if meta_dict["videos"][name]["objects"][obj]["category"] not in category_list:
                    category_list.append(meta_dict["videos"][name]["objects"][obj]["category"])
        category_list = sorted(list(OrderedDict.fromkeys(category_list)))
        category_ditct = {k: v+1 for v, k in enumerate(category_list)} ## zero is always for the background
    return category_ditct



def map_instances(data, meta, category_dict):
    bs, fs, h, w = data.shape
    for i, datum in enumerate(data):
        for j, frame in enumerate(data):
            objects = torch.unique(frame)
            for k, obj in enumerate(objects):
                if int(obj.item()) == 0:
                    continue
                frame[frame == obj] = category_dict[meta[str(int(obj.item()))]["category"]]
    return data


class VideoDataset(torch.utils.data.Dataset):
    ## The data loader gets training sample and annotations direcotories, sampling mode, number of clips that is being sampled of each training video, number of frames in each clip
    ## and number of labels for each training clip. 
    ## Note that the number of annotations should be exactly similar to the number of frames existing in the training path.
    ## Frame_transform is a function that transforms the frames of the video. It is applied to each frame of the video.
    ## Target_transform is a function that transforms the annotations of the video. It is applied to each annotation of the video.
    ## Video_transform is a function that transforms the whole video. It is applied to both frames and annotations of the video.
    ## The same set of transformations is applied to the clips of the video.

    def __init__(self, classes_directory, annotations_directory, sampling_mode, num_clips, num_frames, frame_transform=None, target_transform=None, video_transform=None, meta_file_directory=None, regular_step=1) -> None:
        super().__init__()
        self.train_dict = get_file_path(classes_directory)
        self.train_dict_lenghts = {}
        self.find_directory_length()
        if (annotations_directory != "") and (os.path.exists(annotations_directory)):
            self.train_annotations_dict = get_file_path(annotations_directory)
            self.use_annotations = True
        else:
            self.use_annotations = False
            print("Because there is no annotation directory, only training samples have been loaded.")
        if (meta_file_directory is not None):
            if (os.path.exists(meta_file_directory)):
                print("Meta file has been read.")
                file = open(meta_file_directory)
                self.meta_dict = json.load(file)
            else:
                self.meta_dict = None
                print("There is no meta file.")
        else:
            print("Meta option is off.")
            self.meta_dict = None
         
        self.sampling_mode = sampling_mode
        self.num_clips = num_clips
        self.num_frames = num_frames
        self.frame_transform = frame_transform
        self.target_transform = target_transform
        self.video_transform = video_transform
        self.regular_step = regular_step
        self.keys = list(self.train_dict.keys())
        if self.use_annotations:
            self.annotation_keys = list(self.train_annotations_dict.keys())
        
    def __len__(self):
        return len(self.keys)

    
    def find_directory_length(self):
        for key in self.train_dict:
            self.train_dict_lenghts[key] = len(self.train_dict[key])

    
    def read_clips(self, path, clip_indices):
        clips = []
        files = sorted(glob.glob(path + "/" + "*.jpg"))
        if len(files) == 0:
            files = sorted(glob.glob(path + "/" + "*.png"))
        for i in range(len(clip_indices)):
            images = []
            for j in clip_indices[i]:
                # frame_path = path + "/" + f'{j:05d}' + ".jpg"
                frame_path = files[j]
                if not os.path.exists(frame_path):
                    frame_path = path + "/" + f'{j:05d}' + ".png"
                if not os.path.exists(frame_path): ## This is for kinetics dataset
                    frame_path = path + "/" + f'img_{(j + 1):05d}' + ".jpg" 
                if not os.path.exists(frame_path): ## This is for kinetics dataset
                    frame_path = path + "/" + f'frame_{(j + 1):010d}' + ".jpg" 

                images.append(Image.open(frame_path))
            clips.append(images)
        return clips
    
    
    def generate_indices(self, size, sampling_num):
        indices = []
        for i in range(self.num_clips):
            if self.sampling_mode == SamplingMode.UNIFORM:
                    if size < sampling_num:
                        ## sample repeatly
                        idx = random.choices(range(0, size), k=sampling_num)
                    else:
                        idx = random.sample(range(0, size), sampling_num)
                    idx.sort()
                    indices.append(idx)
            elif self.sampling_mode == SamplingMode.DENSE:
                    base = random.randint(0, size - sampling_num)
                    idx = range(base, base + sampling_num)
                    indices.append(idx)
            elif self.sampling_mode == SamplingMode.Full:
                    indices.append(range(0, size))
            elif self.sampling_mode == SamplingMode.Regular:
                if size < sampling_num * self.regular_step:
                    step = size // sampling_num
                else:
                    step = self.regular_step
                base = random.randint(0, size - (sampling_num * step))
                idx = range(base, base + (sampling_num * step), step)
                indices.append(idx)
        return indices
    

    def read_batch(self, path, annotation_path=None, frame_transformation=None, target_transformation=None, video_transformation=None):
        size = self.train_dict_lenghts[path]
        # sampling_num = size if self.num_frames > size else self.num_frames
        clip_indices = self.generate_indices(self.train_dict_lenghts[path], self.num_frames)
        sampled_clips = self.read_clips(path, clip_indices)
        annotations = []
        sampled_clip_annotations = []
        if annotation_path is not None:
            sampled_clip_annotations = self.read_clips(annotation_path, clip_indices)
            if target_transformation is not None:
                for i in range(len(sampled_clip_annotations)):
                    sampled_clip_annotations[i] = target_transformation(sampled_clip_annotations[i])
        if frame_transformation is not None:
            for i in range(len(sampled_clips)):
                try:
                    sampled_clips[i] = frame_transformation(sampled_clips[i])
                except:
                    print("Error in frame transformation")
        if video_transformation is not None:
            for i in range(len(sampled_clips)):
                if len(sampled_clip_annotations) != 0:
                    sampled_clips[i], sampled_clip_annotations[i] = video_transformation(sampled_clips[i], sampled_clip_annotations[i])
                else:
                    sampled_clips[i] = video_transformation(sampled_clips[i])
        sampled_data = torch.stack(sampled_clips)
        if len(sampled_clip_annotations) != 0:
            sampled_annotations = torch.stack(sampled_clip_annotations)
            if sampled_annotations.size(0) != 0:
                sampled_annotations = (255 * sampled_annotations).type(torch.uint8) 
                if sampled_annotations.shape[2] == 1:
                    sampled_annotations = sampled_annotations.squeeze(2)
        else:
            sampled_annotations = torch.empty(0)
        ## squeezing the annotations to be in the shape of (num_sample, num_clips, num_frames, height, width)
        return sampled_data, sampled_annotations


    def read_batch_with_new_transforms(self, path, annotation_path=None, frame_transformation=None, target_transformation=None, video_transformation=None):
        size = self.train_dict_lenghts[path]
        # sampling_num = size if self.num_frames > size else self.num_frames
        clip_indices = self.generate_indices(self.train_dict_lenghts[path], self.num_frames)
        sampled_clips = self.read_clips(path, clip_indices)
        annotations = []
        sampled_clip_annotations = []
        labels_dict = {}
        if annotation_path is not None:
            sampled_clip_annotations = self.read_clips(annotation_path, clip_indices)
            if target_transformation is not None:
                for i in range(len(sampled_clip_annotations)):
                    sampled_clip_annotations[i] = target_transformation(sampled_clip_annotations[i])
        if frame_transformation is not None:
            for i in range(len(sampled_clips)):
                # try:
                multi_crops, labels_dict = frame_transformation(sampled_clips[i])
                # except:
                #     print("Error in frame transformation")
        if video_transformation is not None:
            for i in range(len(sampled_clips)):
                if len(sampled_clip_annotations) != 0:
                    sampled_clips[i], sampled_clip_annotations[i] = video_transformation(sampled_clips[i], sampled_clip_annotations[i])
                else:
                    sampled_clips[i] = video_transformation(sampled_clips[i])
        # sampled_data = torch.stack(sampled_clips)
        if len(sampled_clip_annotations) != 0:
            sampled_annotations = torch.stack(sampled_clip_annotations)
            if sampled_annotations.size(0) != 0:
                sampled_annotations = (255 * sampled_annotations).type(torch.uint8) 
                if sampled_annotations.shape[2] == 1:
                    sampled_annotations = sampled_annotations.squeeze(2)
        else:
            sampled_annotations = torch.empty(0)
        ## squeezing the annotations to be in the shape of (num_sample, num_clips, num_frames, height, width)
        return multi_crops, labels_dict, sampled_annotations
    

    def __getitem__(self, idx): ### (B, num_clips, num_frames, C, H, W) returns None if the annotation flag is off. Be careful when loading the data.
        # idx = 0  ## This is a hack to make the code work with the dataloader.
        # idx = random.randint(0, 5)
        video_path = self.keys[idx]
        dir_name = video_path.split("/")[-1]
        annotations = None
        annotations_path = None
        if (self.use_annotations):
            annotations_path = self.annotation_keys[idx]
            # annotations = self.read_annotations(annotations_path, self.target_transform, indices=indices)
        data, annotations = self.read_batch(video_path, annotations_path, self.frame_transform, self.target_transform ,self.video_transform)   
        if self.meta_dict is not None:
            category_dict = make_categories_dict(self.meta_dict, "davis")
            meta_dict = self.meta_dict["videos"][dir_name]["objects"]
            annotations = map_instances(annotations, meta_dict, category_dict)

        return data, annotations


def save_mapping(mapping, filepath="zip_mapping.json"):
    with open(filepath, "w") as f:
        json.dump(mapping, f)

def load_mapping(filepath="zip_mapping.json"):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None

from collections import defaultdict

def locate_and_load_set_lists(zip_mapping_path):
    # Load existing zip mapping
    with open(zip_mapping_path, 'r') as f:
        zip_mapping = json.load(f)

    category_data = defaultdict(dict)

    for category, zip_files in zip_mapping.items():
        print(f"Processing category '{category}' with {len(zip_files)} zip files")
        
        # Search for 'set_lists' files within each zip for the category
        for zip_file in zip_files:
            with ZipFile(zip_file, 'r') as z:
                # Check for the presence of "set_lists/set_lists_manyview_train.json"
                set_list_files = [f for f in z.namelist() if "set_lists/set_lists_manyview_dev_0.json" in f]
                
                if set_list_files:
                    # Load the first match (assuming only one relevant file per zip)
                    with z.open(set_list_files[0]) as f:
                        data = json.load(f)

                        if "train_known" in data:
                            print(f"Found 'train_known' entries in {set_list_files[0]} for category '{category}'")
                            
                            # Store each entry in category_data for easier access
                            for entry in data["train_known"]:
                                sequence, frame_idx, img_path = entry
                                
                                # Initialize sequence if it doesn't exist
                                if sequence not in category_data[category]:
                                    category_data[category][sequence] = []
                                    
                                # Append frame information (index and path)
                                category_data[category][sequence].append((frame_idx, img_path))
        
        # Ensure frames in each sequence are sorted by frame index
        for sequence in category_data[category]:
            category_data[category][sequence].sort()  # Sort by frame_idx
            print(f"Sorted frames for category '{category}', sequence '{sequence}': {category_data[category][sequence]}")
    
    # Save the structured mapping to avoid recomputation
    detailed_mapping_path = os.path.join("/home/isimion1/timet/Timetuning_v2/detailed_mapping.json")
    #detailed_mapping_path = os.path.join(root_directory, "detailed_mapping.json")
    with open(detailed_mapping_path, 'w') as f:
        json.dump(category_data, f)

    print("Finished creating detailed mapping.")
    return category_data
    

class CO3DDataset(Dataset):

    def __init__(self, root_directory, subset_name, sampling_mode, num_clips, num_frames,
             frame_transform=None, target_transform=None, video_transform=None, regular_step=1):
        super().__init__()
        self.root_directory = root_directory
        self.subset_name = subset_name
        self.sampling_mode = sampling_mode
        self.num_clips = num_clips
        self.num_frames = num_frames
        self.frame_transform = frame_transform
        self.target_transform = target_transform
        self.video_transform = video_transform
        self.regular_step = regular_step

        # Load the precomputed detailed mapping
        mapping_path = os.path.join("/home/isimion1/timet/Timetuning_v2/zip_mapping.json")
        detailed_mapping_path = os.path.join("/home/isimion1/timet/Timetuning_v2/detailed_mapping.json")
        self.category_data = load_mapping(detailed_mapping_path)
        if self.category_data is None:
            print("Creating detailed mapping for the first time...")
            self.category_data = locate_and_load_set_lists(mapping_path)
            save_mapping(self.category_data, mapping_path)
        else:
            
            print("Loaded existing detailed mapping.")

        # Flatten the mapping to create sequences and frames lists
        self.sequences = [
            {"category": category, "sequence": sequence, "frames": frames}
            for category, seq_data in self.category_data.items()
            for sequence, frames in seq_data.items()
        ]

        print("Number of sequences:", len(self.sequences))

    
    def organize_zips_by_category(self):
        category_zip_map = {}
        for zip_path in glob.glob(os.path.join(self.root_directory, "*.zip")):
            category = os.path.basename(zip_path).split("_")[0]
            if category not in category_zip_map:
                category_zip_map[category] = []
            category_zip_map[category].append(zip_path)
        #print("Category zip map:", category_zip_map)  
        return category_zip_map

    def load_metadata_from_zip(self, zip_file, metadata_filename):
        with ZipFile(zip_file, 'r') as z:
            if metadata_filename in z.namelist():
                with z.open(metadata_filename) as f:
                    with gzip.open(f, 'rt', encoding='utf-8') as gf:
                        print(f'Metadata file {metadata_filename} found in {zip_file}.')
                        return json.load(gf)
        return None

    def get_file_paths_from_zips(self):
        folder_file_path = {}
        
        for category, zip_files in self.category_zip_map.items():
            images = []
            frame_annotations = None
            sequence_annotations = None

            for zip_file in zip_files:
                with ZipFile(zip_file, 'r') as z:
                    # Collect images with their zip file location
                    all_files = z.namelist()
                    img_files = [f for f in all_files if f.startswith(f"{category}/") and "images/" in f and f.endswith((".jpg", ".png"))]

                    if img_files:
                        images.extend((img, zip_file) for img in img_files)  # Store tuple of image path and zip file path

                    # Load annotations if not already loaded
                    if frame_annotations is None:
                        frame_annotations = self.load_metadata_from_zip(zip_file, "frame_annotations.jgz")
                    if sequence_annotations is None:
                        sequence_annotations = self.load_metadata_from_zip(zip_file, "sequence_annotations.jgz")

            images.sort(key=lambda x: x[0])  # Sort by image path
            if images:
                folder_file_path[category] = {
                    'images': images,  # List of (image_path, zip_file_path) tuples
                    'frame_annotations': frame_annotations,
                    'sequence_annotations': sequence_annotations,
                }
        
        return folder_file_path



    def __len__(self):
        return len(self.sequences)

    def generate_indices(self, size):
        indices = []
        for _ in range(self.num_clips):
            if self.sampling_mode == SamplingMode.UNIFORM:
                idx = random.sample(range(size), self.num_frames) if size >= self.num_frames else random.choices(range(size), k=self.num_frames)
            elif self.sampling_mode == SamplingMode.DENSE:
                base = random.randint(0, size - self.num_frames)
                idx = list(range(base, base + self.num_frames))
            elif self.sampling_mode == SamplingMode.Regular:
                step = min(self.regular_step, size // self.num_frames) if size >= self.num_frames * self.regular_step else size // self.num_frames
                idx = list(range(0, size, step))[:self.num_frames]
            indices.append(idx)
        return indices

    def read_clips(self, clip_indices, image_paths):
        clips = []

        for indices in clip_indices:
            images = []
            for idx in indices:
                img_path, zip_file = image_paths[idx]  # Retrieve image path and corresponding zip file
                try:
                    with ZipFile(zip_file, 'r') as z:
                        images.append(Image.open(z.open(img_path)).convert("RGB"))
                except KeyError:
                    print(f"Warning: File {img_path} not found in {zip_file}. Skipping.")
                    continue  # Skip if the image isn't found in the expected zip

            if images:
                clips.append(images)
            else:
                print(f"Warning: No images found for the clip at indices {indices}. Skipping this clip.")

        return clips


    def get_annotations_for_frame(self, category, frame_number):
        # Fetch metadata for a specific frame within a category if available
        frame_annotations = self.train_dict[category].get('frame_annotations')
        if frame_annotations:
            for frame in frame_annotations:
                if frame['frame_number'] == frame_number:
                    return frame
        return None

    def read_batch(self, category):
        # Reads a batch of frames and corresponding annotations from multiple zip files in a category
        zip_files = self.train_dict[category]['zip_files']
        image_paths = self.train_dict[category]['images']
        size = len(image_paths)
        
        clip_indices = self.generate_indices(size)
        clips = self.read_clips(zip_files, clip_indices, image_paths)
        
        # Apply transformations
        if self.frame_transform:
            clips = [self.frame_transform(clip) for clip in clips]
        
        if self.video_transform:
            clips = [self.video_transform(clip) for clip in clips]
        
        data = torch.stack([torch.stack([torch.from_numpy(np.array(img)).permute(2, 0, 1) for img in clip]) for clip in clips])

        return data, None  

   
    def __getitem__(self, idx):
        category, sequence, frames = self.sequences[idx]
        if len(frames) >= self.num_frames:
            start = random.randint(0, len(frames) - self.num_frames)
            selected_frames = frames[start:start + self.num_frames]
        else:
            selected_frames = random.choices(frames, k=self.num_frames)

        images = []
        for frame_idx, img_path in selected_frames:
            zip_file_path = os.path.join(self.root_directory, f"{category}.zip")
            with ZipFile(zip_file_path, 'r') as z:
                img = Image.open(z.open(img_path)).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                images.append(img)

        images_tensor = torch.stack([torch.from_numpy(np.array(img)).permute(2, 0, 1) for img in images])
        return images_tensor


CLASS_IDS = {
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "train": 19,
    "tvmonitor": 20,
}


class SPairDataset(torch.utils.data.Dataset):
    r"""Inherits CorrespondenceDataset"""

    def __init__(
        self,
        root,
        split='test',
        image_size=224,
        image_mean="imagenet",
        use_bbox=True,
        class_name=None,
        num_instances=None,
        vp_diff=None,
    ):
        """
        Constructs the SPair Dataset loader

        Inputs:
            root: Dataset root (where SPair is found; kinda odd TODO)
            thresh: how the threshold is calculated [img, bbox]
            split: dataset split to be used
            task: task for this dataset
        """
        super().__init__()
        assert split in ["train", "valid", "test"]
        print('starting to get SPair')
        self.root = root
        self.split = split
        self.image_size = image_size
        self.use_bbox = use_bbox

        if image_mean == "clip":
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        elif image_mean == "imagenet":
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            raise ValueError()

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.NEAREST,
                    antialias=True,
                ),
                transforms.ToTensor(),
            ]
        )

        instances = self.get_pair_annotations()

        if class_name:
            c_insts = [_a for _a in instances if _a["category"] == class_name]
            instances = c_insts

        if vp_diff is not None:
            instances = [_a for _a in instances if _a["viewpoint_variation"] == vp_diff]

        if num_instances:
            random.seed(20)
            random.shuffle(instances)
            instances = instances[:num_instances]

        self.instances = instances
        self.image_annotations = self.get_image_annotations()

    def process_keypoints(self, kp_dict, bbox, num_kps=None):
        num_kps = len(kp_dict) if num_kps is None else num_kps
        all_kps = [(kp_dict[str(i)], i) for i in range(num_kps) if kp_dict[str(i)]]
        kps_xy = torch.tensor([_xy for _xy, _ in all_kps]).int()
        kps_id = torch.tensor([_id for _, _id in all_kps]).long()

        if bbox:
            kps_xy[:, 0] -= bbox[0]
            kps_xy[:, 1] -= bbox[1]

        # generate full tensor
        kps = torch.zeros(num_kps, 3).int()
        kps[kps_id, :2] = kps_xy
        kps[kps_id, 2] = 1

        return kps

    def __getitem__(self, index, square=True):
        pair_i = self.instances[index]
        print(pair_i)
        class_name = pair_i["category"]
        class_dict = self.image_annotations[class_name]
        _, view_i, view_j = pair_i["filename"].split(":")[0].split("-")

        # gett bounding boxes
        bbx_i = pair_i["src_bndbox"] if self.use_bbox else None
        bbx_j = pair_i["trg_bndbox"] if self.use_bbox else None

        kps_i = self.process_keypoints(class_dict[view_i]["kps"], bbx_i)
        kps_j = self.process_keypoints(class_dict[view_j]["kps"], bbx_j)

        img_i = self.get_image(class_name, view_i, bbox=bbx_i, square=square)
        seg_i = self.get_mask(class_name, view_i, bbox=bbx_i, square=square)
        img_j = self.get_image(class_name, view_j, bbx_j, square=square)
        seg_j = self.get_mask(class_name, view_j, bbox=bbx_j, square=square)

        # transform image
        hw_i = img_i.size[0]
        hw_j = img_j.size[0]

        if not self.use_bbox:
            l, u, r, d = pair_i["trg_bndbox"]
            max_bbox = max(r - l, d - u)
            max_idim = max(pair_i["trg_imsize"][:2])
            thresh_scale = float(max_bbox) / max_idim
        else:
            thresh_scale = 1.0

        # transform images
        img_i = self.image_transform(img_i)
        img_j = self.image_transform(img_j)
        seg_i = self.mask_transform(seg_i)
        seg_j = self.mask_transform(seg_j)
        kps_i[:, :2] = kps_i[:, :2] * self.image_size / hw_i
        kps_j[:, :2] = kps_j[:, :2] * self.image_size / hw_j

        return img_i, seg_i, kps_i, img_j, seg_j, kps_j, thresh_scale, class_name

    def __len__(self):
        return len(self.instances)

    def get_image(self, class_name, image_name, bbox=None, square=False):
        rel_path = f"JPEGImages/{class_name}/{image_name}.jpg"
        path = os.path.join(self.root, rel_path)

        print(f'path for JPEGImages when loading SPair: {path}')
        with Image.open(path) as f:
            image = np.array(f)

        if bbox:
            l, u, r, d = bbox
            # if square:
            #     max_hw = max(d-u, r-l)
            #     h, w, _ = image.shape
            #     d = min(u + max_hw, h)
            #     r = min(l + max_hw, w)

            image = image[u:d, l:r]

        if square:
            h, w, _ = image.shape
            max_hw = max(h, w)
            image = np.pad(
                image, ((0, max_hw - h), (0, max_hw - w), (0, 0)), constant_values=255
            )

        return Image.fromarray(image)

    def get_mask(self, class_name, image_name, bbox=None, square=False):
        rel_path = f"Segmentation/{class_name}/{image_name}.png"
        path = os.path.join(self.root, rel_path)

        with Image.open(path) as img:
            image = np.array(img)

        if bbox:
            l, u, r, d = bbox
            image = image[u:d, l:r]

        if square:
            h, w = image.shape
            max_hw = max(h, w)
            image = np.pad(image, ((0, max_hw - h), (0, max_hw - w)))

        # big assumption of no other same class within bbox (or image)
        class_id = CLASS_IDS[class_name]
        image = (image == class_id).astype(float) * 255

        return Image.fromarray(image)

    def get_pair_annotations(self):
        split_names = {"train": "trn", "valid": "val", "test": "test"}
        split = split_names[self.split]

        annot_path = os.path.join(self.root, "PairAnnotation", split)
        annot_files = glob.glob(os.path.join(annot_path, "*.json"))
        annots = [json.load(open(_path)) for _path in annot_files]
        return annots

    def get_image_annotations(self):
        annot_path = os.path.join(self.root, "ImageAnnotation")
        classes = os.listdir(annot_path)

        image_annots = {_c: {} for _c in classes}

        for _cls in classes:
            annot_files = glob.glob(os.path.join(annot_path, f"{_cls}/*.json"))
            annots = [json.load(open(_path)) for _path in annot_files]
            annots = {_a["filename"].split(".")[0]: _a for _a in annots}
            image_annots[_cls] = annots

        return image_annots
    
class YVOSDataset(VideoDataset):

    def __init__(self, classes_directory, annotations_directory, sampling_mode, num_clips, num_frames, frame_transform=None, target_transform=None, video_transform=None, meta_file_directory=None, regular_step=1) -> None:
        super().__init__(classes_directory, annotations_directory, sampling_mode, num_clips, num_frames, frame_transform, target_transform, video_transform, meta_file_directory, regular_step=regular_step)
        if self.meta_dict is not None:
            self.category_dict = make_categories_dict(self.meta_dict, "ytvos")

    def __getitem__(self, idx): ### (B, num_clips, num_frames, C, H, W) returns None if the annotation flag is off. Be careful when loading the data.
        video_path = self.keys[idx]
        dir_name = video_path.split("/")[-1]
        annotations = None
        annotations_path = None
        if (self.use_annotations):
            annotations_path = self.annotation_keys[idx]
            # annotations = self.read_annotations(annotations_path, self.target_transform, indices=indices)
        data, annotations = self.read_batch(video_path, annotations_path, self.frame_transform, self.target_transform ,self.video_transform)   
        if self.meta_dict is not None:
            annotations = map_instances(annotations, self.meta_dict["videos"][dir_name]["objects"], self.category_dict)

        # else:
            # annotations = convert_to_indexed_RGB(annotations)
        return data, annotations



class TimeTYVOSDataset(VideoDataset):

    def __init__(self, classes_directory, annotations_directory, sampling_mode, num_clips, num_frames, frame_transform=None, target_transform=None, video_transform=None, meta_file_directory=None, regular_step=1) -> None:
        super().__init__(classes_directory, annotations_directory, sampling_mode, num_clips, num_frames, frame_transform, target_transform, video_transform, meta_file_directory, regular_step=regular_step)
        if self.meta_dict is not None:
            self.category_dict = make_categories_dict(self.meta_dict, "timetytvos")

    def __getitem__(self, idx): ### (B, num_clips, num_frames, C, H, W) returns None if the annotation flag is off. Be careful when loading the data.
        video_path = self.keys[idx]
        dir_name = video_path.split("/")[-1]
        annotations = None
        annotations_path = None
        if (self.use_annotations):
            annotations_path = self.annotation_keys[idx]
            # annotations = self.read_annotations(annotations_path, self.target_transform, indices=indices)
        data, labels, annotations = self.read_batch_with_new_transforms(video_path, annotations_path, self.frame_transform, self.target_transform ,self.video_transform)   
        if self.meta_dict is not None:
            annotations = map_instances(annotations, self.meta_dict["videos"][dir_name]["objects"], self.category_dict)

        # else:
            # annotations = convert_to_indexed_RGB(annotations)
        return data, labels, annotations


class Kinetics(VideoDataset):

    def __init__(self, classes_directory, sampling_mode, num_clips, num_frames, frame_transform=None, target_transform=None, video_transform=None, meta_file_directory=None, regular_step=1) -> None:
        super().__init__(classes_directory, "", sampling_mode, num_clips, num_frames, frame_transform, target_transform, video_transform, meta_file_directory, regular_step=regular_step)

    def __getitem__(self, idx): ### (B, num_clips, num_frames, C, H, W) returns None if the annotation flag is off. Be careful when loading the data.
        video_path = self.keys[idx]
        dir_name = video_path.split("/")[-1]
        annotations = None
        annotations_path = None
        data, annotations = self.read_batch_with_new_transforms(video_path, None, self.frame_transform, self.target_transform ,self.video_transform)   
        if self.meta_dict is not None:
            annotations = map_instances(annotations, self.meta_dict["videos"][dir_name]["objects"], self.category_dict)

        # else:
            # annotations = convert_to_indexed_RGB(annotations)
        return data, annotations


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
        print(f'seg dir is {seg_dir}')
        image_dir = os.path.join(root, 'images')

        print(f'img dir is {image_dir}')
        print(f'root dir is {root}')
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

    def __init__(self, batch_size, train_transform, val_transform, test_transform,  dir="/scratch-shared/isimion1/pascal/VOCSegmentation", num_workers=0) -> None:
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
        print(f"Train size : {len(self.train_dataset)}")
        print(f"Val size : {len(self.val_dataset)}")

    def get_train_dataloader(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
    
    def get_val_dataloader(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    
    def get_test_dataloader(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    
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
    


class VideoDataModule():

    def __init__(self, name, path_dict, num_clips, num_clip_frames, sampling_mode, regular_step, batch_size, num_workers=0, world_size=1, rank=0):
        super().__init__()
        self.name = name
        self.path_dict = path_dict
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.class_directory = self.path_dict["class_directory"]
        self.annotations_directory = self.path_dict["annotation_directory"]
        self.meta_file_path = self.path_dict["meta_file_path"]
        self.num_clip_frames = num_clip_frames
        self.sampling_mode = sampling_mode
        self.regular_step = regular_step
        self.num_clips = num_clips
        self.world_size = world_size
        self.rank = rank
        self.sampler = None
        self.data_loader = None
        self.subset_name = "train"
        print("Class Directory:", self.class_directory)
        print("Annotations Directory:", self.annotations_directory)
        print("Meta File Path:", self.meta_file_path)


    
    def setup(self, transforms_dict):
        data_transforms = transforms_dict["data_transforms"]
        target_transforms = transforms_dict["target_transforms"]
        shared_transforms = transforms_dict["shared_transforms"]
        if self.name == "timetytvos":
            self.dataset = TimeTYVOSDataset(self.class_directory, self.annotations_directory, self.sampling_mode, self.num_clips, self.num_clip_frames, data_transforms, target_transforms, shared_transforms, self.meta_file_path, self.regular_step)
        elif self.name == "ytvos":
            self.dataset = YVOSDataset(self.class_directory, self.annotations_directory, self.sampling_mode, self.num_clips, self.num_clip_frames, data_transforms, target_transforms, shared_transforms, self.meta_file_path, self.regular_step)
        elif self.name == "kinetics":
            self.dataset = Kinetics(self.class_directory, self.annotations_directory, self.sampling_mode, self.num_clips, self.num_clip_frames, data_transforms, target_transforms, shared_transforms, self.meta_file_path, self.regular_step)
        elif self.name == "co3d":
            print('using CO3D dataset')
            self.dataset = CO3DDataset(
                root_directory=self.class_directory, 
                subset_name=self.subset_name,  # Define subset as "train", "val", or "test"
                sampling_mode=self.sampling_mode, 
                num_clips=self.num_clips, 
                num_frames=self.num_clip_frames, 
                frame_transform=data_transforms, 
                target_transform=target_transforms, 
                video_transform=shared_transforms, 
                regular_step=self.regular_step
            )       
        else:
            self.dataset = VideoDataset(self.class_directory, self.annotations_directory, self.sampling_mode, self.num_clips, self.num_clip_frames, data_transforms, target_transforms, shared_transforms, self.meta_file_path, self.regular_step)
        print(f"Dataset size : {len(self.dataset)}")
    
    def make_data_loader(self, shuffle=True):
        if self.world_size > 1:
            self.sampler = DistributedSampler(self.dataset, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle)
            self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, sampler=self.sampler, drop_last=True)
        else:
            self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, pin_memory=True, drop_last=True)
    
    def get_data_loader(self):
        return self.data_loader


class CocoDetection(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transform is not None:
            image = self.transform(image)
            target = torch.Tensor([0])

        elif self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)

    


class CocoDataModule():
    def __init__(self, batch_size, train_transform, val_transform, test_transform,  img_dir="", annotation_dir="", num_workers=0) -> None:
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
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
        self.train_dataset = CocoDetection(self.img_dir, self.annotation_dir, transform=self.image_train_transform)
        self.val_dataset = CocoDetection(self.img_dir, self.annotation_dir, transforms=self.shared_val_transform)
        self.test_dataset = CocoDetection(self.img_dir, self.annotation_dir, transforms=self.shared_test_transform)
        print(f"Train size : {len(self.train_dataset)}")
        print(f"Val size : {len(self.val_dataset)}")

    def get_train_dataloader(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
    
    def get_val_dataloader(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    
    def get_test_dataloader(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    
    def get_train_dataset_size(self):
        return len(self.train_dataset)
    
    def get_val_dataset_size(self):
        return len(self.val_dataset)
    
    def get_test_dataset_size(self):
        return len(self.test_dataset)
    
    def get_module_name(self):
        return "CocoDataModule"
    
    def get_num_classes(self):
        return 91
    
    

    


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
        # RandomHorizontalFlip(probability=0.1),
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



def test_video_data_module(logger):
    rand_color_jitter = video_transformations.RandomApply([video_transformations.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.8)
    data_transform_list = [rand_color_jitter, video_transformations.RandomGrayscale(), video_transformations.RandomGaussianBlur()]
    data_transform = video_transformations.Compose(data_transform_list)
    video_transform_list = [video_transformations.ClipToTensor()]
    target_transform = video_transformations.Compose(video_transform_list)
    video_transform = video_transformations.TimeTTransform([224, 96], [1, 4], [0.25, 0.05], [1., 0.25], 1, 0.01, 1)
    num_clips = 1
    batch_size = 8
    num_workers = 4
    num_clip_frames = 4
    regular_step = 1
    transformations_dict = {"data_transforms": video_transform, "target_transforms": target_transform, "shared_transforms": None}
    prefix = "/ssdstore/ssalehi/dataset"
    data_path = os.path.join(prefix, "train1/JPEGImages/")
    annotation_path = os.path.join(prefix, "train1/Annotations/")
    meta_file_path = os.path.join(prefix, "train1/meta.json")
    path_dict = {"class_directory": data_path, "annotation_directory": annotation_path, "meta_file_path": meta_file_path}
    sampling_mode = SamplingMode.DENSE
    video_data_module = VideoDataModule("timetytvos", path_dict, num_clips, num_clip_frames, sampling_mode, regular_step, batch_size, num_workers)
    video_data_module.setup(transformations_dict)
    data_loader = video_data_module.data_loader
    logging_directory = "data_loader_log/"

    if os.path.exists(logging_directory):
        os.system(f'rm -r {logging_directory}')
    os.makedirs(logging_directory)

    for i, train_data in enumerate(data_loader):
        datum, labels, annotations = train_data
        print("===========================")
        print("")
        annotations = annotations.squeeze(1)
        datum = datum.squeeze(1)
        datum = denormalize_video(datum)
        print((torch.unique(annotations)))
        print(datum.shape)
        print(annotations.shape)
        # visualize_sampled_videos(datum, "data_loader_log/", f"test_{i}.avi")
        # visualize_sampled_videos(annotations, "data_loader_log/", f"test_anotations_{i}.avi")
        make_seg_maps(datum, annotations, logging_directory, f"test_seg_maps_{i}.avi")



def test_video_dataloader_with_timet_transforms(logger):
    sampling = "dense"
    num_clips = 1
    batch_size = 1
    num_workers = 4
    num_clip_frames = 8
    regular_step = 25
    num_crops = 4
    logging_directory = "data_loader_log/"
    video_transform_list = [video_transformations.RandomResizedCrop((224, 224)), video_transformations.ClipToTensor()] #video_transformations.RandomResizedCrop((224, 224))
    target_transform = video_transformations.Compose(video_transform_list)
    video_transform = video_transformations.TimeTTransform([224, 96], [1, num_crops], [0.35, 0.25], [1., 0.4], 1, 0.01, 1)
    world_size = 1
    transformations_dict = {"data_transforms": video_transform, "target_transforms": target_transform, "shared_transforms": None}
    prefix = "/ssdstore/ssalehi/dataset"
    data_path = os.path.join(prefix, "all_frames/train_all_frames/JPEGImages/")
    annotation_path = "" # os.path.join(prefix, "train1/Annotations/")
    meta_file_path = "" # os.path.join(prefix, "train1/meta.json")
    path_dict = {"class_directory": data_path, "annotation_directory": annotation_path, "meta_file_path": meta_file_path}
    if sampling == "dense":
        sampling_mode = SamplingMode.DENSE
    elif sampling == "uniform":
        sampling_mode = SamplingMode.UNIFORM
    elif sampling == "full":
        sampling_mode = SamplingMode.FULL
    else:
        raise ValueError("Sampling mode is not valid")
    video_data_module = VideoDataModule("timetytvos", path_dict, num_clips, num_clip_frames, sampling_mode, regular_step, batch_size, num_workers, world_size=world_size)
    video_data_module.setup(transformations_dict)
    video_data_module.make_data_loader()
    data_loader = video_data_module.get_data_loader()
    for i, batch in enumerate(data_loader):
        batch_crop_list, label, annotations = batch
        global_crops_1 = batch_crop_list[0]
        annotations = annotations.squeeze(1)
        datum = global_crops_1
        annotations = annotations.squeeze(1)
        datum = datum.squeeze(1)
        # datum = denormalize_video(datum)
        print((torch.unique(annotations)))
        print(datum.shape)
        print(annotations.shape)
        visualize_sampled_videos(datum[0], "data_loader_log/", f"test_{i}.avi")
        local_crop = batch_crop_list[1]
        local_crop = local_crop.squeeze(1)
        print(local_crop.shape)
        # local_crop = denormalize_video(local_crop)
        visualize_sampled_videos(local_crop[0], "data_loader_log/", f"test_local_{i}.avi")
        # visualize_sampled_videos(annotations, "data_loader_log/", f"test_anotations_{i}.avi")
        # make_seg_maps(datum, annotations, logging_directory, f"test_seg_maps_{i}.avi")
    

if __name__ == "__main__":
    ## init wandb
    logger = wandb.init(project=project_name, group="data_loader", tags="PascalVOCDataModule", job_type="eval")
    ## test data module
    # test_pascal_data_module(logger)
    ## finish wandb
    # logger.finish()

    # get_file_path("/ssdstore/ssalehi/dataset/val1/JPEGImages/")
    test_video_dataloader_with_timet_transforms(logger)
