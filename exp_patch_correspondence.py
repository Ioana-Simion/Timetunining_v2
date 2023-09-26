import argparse
from datetime import date
import os
from sqlite3 import Time
import time
import torch
from torchvision import transforms
from pytorchvideo.data import Ucf101, make_clip_sampler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from clustering import PerDatasetClustering
from data_loader import Cifar10_Handler, PascalVOCDataModule, SamplingMode, VideoDataModule
from eval_metrics import PredsmIoU
from evaluator import LinearFinetuneModule
from models import CrossAttentionBlock, FeatureExtractor
from my_utils import find_optimal_assignment, overlay, denormalize_video
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Lambda,
    RandomCrop,
    RandomHorizontalFlip, 
)
import wandb
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from optimizer import PatchCorrespondenceOptimizer
import torchvision.transforms as trn

from image_transformations import Compose, Resize
import video_transformations

project_name = "TimeTuning_v2"
## generate ListeColorMap of distinct colors

## what are the colors for red, blue, green, brown, yello, orange, purple, white, black, maroon, olive, teal, navy, gray, silver
## Fill the ListedColormap with the colors above

cmap = ListedColormap(['#FF0000', '#0000FF', '#008000', '#A52A2A', '#FFFF00', '#FFA500', '#800080', '#FFFFFF', '#000000', '#800000', '#808000', '#008080', '#000080', '#808080', '#C0C0C0'])

def generate_random_crop(img, crop_size):
    ## generate a random crop mask with all 1s on the img with  crop_scale=(0.05, 0.3), and size crop_size
    bs, c, h, w = img.shape
    crop = torch.zeros((bs, h, w))
    x = torch.randint(0, h - crop_size, (1,)).item()
    y = torch.randint(0, w - crop_size, (1,)).item()
    crop[:, x:x + crop_size, y:y + crop_size] = 1
    return crop


def random_crop_mask(img, aspect_ratio_range=(3/4, 4/3), scale_range=(0.05, 0.3), mask_height=None, mask_width=None):
    """
    Generate a random crop mask with a given aspect ratio.
    
    H, W: Image dimensions
    aspect_ratio: Desired aspect ratio = mask_width/mask_height
    mask_height or mask_width: Specify one of them and the other will be computed using aspect_ratio.
    """
    bs, c, H, W = img.shape
    # Extract values from the provided ranges
    min_aspect, max_aspect = aspect_ratio_range
    min_scale, max_scale = scale_range
    
    # Randomly select an aspect ratio and scale within the provided range
    random_aspect_ratio = torch.rand(1).item() * (max_aspect - min_aspect) + min_aspect
    random_scale = torch.rand(1).item() * (max_scale - min_scale) + min_scale
    
    # Compute mask dimensions based on random scale and aspect ratio
    mask_width = int(W * random_scale * random_aspect_ratio)
    mask_height = int(W * random_scale / random_aspect_ratio)

    # Check if mask dimensions exceed the image dimensions
    if mask_height > H or mask_width > W:
        raise ValueError("Mask dimensions exceed the image dimensions.")

    # Generate a random starting point
    y = torch.randint(0, H - mask_height + 1, (1,)).item()
    x = torch.randint(0, W - mask_width + 1, (1,)).item()

    # Create the mask with all zeros and set the crop area to ones
    mask = torch.zeros((bs, H, W))
    mask[:, y:y+mask_height, x:x+mask_width] = 1

    return mask

## a function that generates random crop masks for a batch of images
def generate_random_crop_masks(imgs, aspect_ratio_range=(3/4, 4/3), scale_range=(0.05, 0.3), mask_height=None, mask_width=None):

    bs, c, H, W = imgs.shape
    # Extract values from the provided ranges
    min_aspect, max_aspect = aspect_ratio_range
    min_scale, max_scale = scale_range

    # Randomly select an aspect ratio and scale within the provided range
    random_aspect_ratio = torch.rand(1).item() * (max_aspect - min_aspect) + min_aspect
    random_scale = torch.rand(1).item() * (max_scale - min_scale) + min_scale

    # Compute mask dimensions based on random scale and aspect ratio
    mask_width = int(W * random_scale * random_aspect_ratio)
    mask_height = int(W * random_scale / random_aspect_ratio)

    # Check if mask dimensions exceed the image dimensions
    if (mask_height > H) or (mask_width > W):
        raise ValueError("Mask dimensions exceed the image dimensions.")
    

    y_uniform = torch.rand(bs).to(imgs.device)
    x_uniform = torch.rand(bs).to(imgs.device)
    
    # Scale and shift to the desired range [0, H - mask_height[i] + 1) for each i
    y = (y_uniform * (H - mask_height + 1)).floor().long()
    x = (x_uniform * (W - mask_width + 1)).floor().long()

    x = (x // 16) * 16
    y = (y // 16) * 16

    # Create the mask with all zeros and set the crop area to ones
    mask = torch.zeros((bs, H, W)).to(imgs.device)
    for i in range(bs):
        mask[i, y[i]:y[i]+mask_height, x[i]:x[i]+mask_width] = 1

    return mask



class CorrespondenceDetection():
    def __init__(self, window_szie, spatial_resolution=14, output_resolution=96) -> None:
        self.window_size = window_szie
        self.neihbourhood = self.restrict_neighborhood(spatial_resolution, spatial_resolution, self.window_size)
        self.output_resolution = output_resolution

    
    def restrict_neighborhood(self, h, w, size_mask_neighborhood):
        # We restrict the set of source nodes considered to a spatial neighborhood of the query node (i.e. ``local attention'')
        mask = torch.zeros(h, w, h, w)
        for i in range(h):
            for j in range(w):
                for p in range(2 * size_mask_neighborhood + 1):
                    for q in range(2 * size_mask_neighborhood + 1):
                        if i - size_mask_neighborhood + p < 0 or i - size_mask_neighborhood + p >= h:
                            continue
                        if j - size_mask_neighborhood + q < 0 or j - size_mask_neighborhood + q >= w:
                            continue
                        mask[i, j, i - size_mask_neighborhood + p, j - size_mask_neighborhood + q] = 1

        # mask = mask.reshape(h * w, h * w)
        return mask

    def __call__(self, features1, features2, crops):
        with torch.no_grad():
            bs, spatial_resolution, spatial_resolution, d_model = features1.shape
            _, h, w = crops.shape
            patch_size = h // spatial_resolution
            crops = crops.reshape(bs, h // patch_size, patch_size, w // patch_size, patch_size).permute(0, 1, 3, 2, 4)
            crops = crops.flatten(3, 4)
            cropped_feature_mask = crops.sum(-1) > 0 ## size (bs, spatial_resolution, spatial_resolution)
            ## find the idx of the croped features_mask
            features1 = F.normalize(features1, dim=-1)
            features2 = F.normalize(features2, dim=-1)
            similarities = torch.einsum('bxyd,bkzd->bxykz', features1, features2)
            most_similar_features_mask = torch.zeros(bs, spatial_resolution, spatial_resolution)
            revised_crop = torch.zeros(bs, spatial_resolution, spatial_resolution)
            self.neihbourhood = self.neihbourhood.to(features1.device)
            similarities = similarities * self.neihbourhood.unsqueeze(0)
            similarities = similarities.flatten(3, 4)
            most_similar_cropped_patches_list = []
            # for i, crp_feature_mask in enumerate(cropped_feature_mask): 
            crp_feature_mask = cropped_feature_mask[0]
            true_coords  = torch.argwhere(crp_feature_mask)
            min_coords = true_coords.min(0).values
            max_coords = true_coords.max(0).values
            rectangle_shape = max_coords - min_coords + 1
            crop_h, crop_w = rectangle_shape
            most_similar_patches = similarities.argmax(-1)
            most_similar_cropped_patches = most_similar_patches[cropped_feature_mask]
            most_similar_cropped_patches = most_similar_cropped_patches.reshape(bs, crop_h, crop_w)
            # most_similar_cropped_patches = F.interpolate(most_similar_cropped_patches.float().unsqueeze(0).unsqueeze(0), size=(self.output_resolution, self.output_resolution), mode='nearest').squeeze(0).squeeze(0)
            # most_similar_cropped_patches_list.append(most_similar_cropped_patches)

            # for i, similarity in enumerate(similarities):
            #     croped_feature_idx = croped_feature_mask[i].nonzero()
            #     for j, mask_idx in enumerate(croped_feature_idx):
            #         # print(mask_idx)
            #         revised_crop[i, mask_idx[0], mask_idx[1]] = 1
            #         min_x, max_x = max(0, mask_idx[0] - self.window_size), min(spatial_resolution, mask_idx[0] + self.window_size)
            #         min_y, max_y = max(0, mask_idx[1] - self.window_size), min(spatial_resolution, mask_idx[1] + self.window_size)
            #         neiborhood_similarity = similarity[mask_idx[0], mask_idx[1], min_x:max_x, min_y:max_y]
            #         max_value = neiborhood_similarity.max()
            #         indices = (neiborhood_similarity == max_value).nonzero()[0]
            #         label_patch_number = (indices[0] + min_x) * spatial_resolution + (indices[1] + min_y)
            #         most_similar_features_mask[i, mask_idx[0], mask_idx[1]] = label_patch_number
            
            # most_similar_cropped_patches = torch.stack(most_similar_cropped_patches_list)
            revised_crop = cropped_feature_mask.float() 
            return most_similar_cropped_patches, revised_crop
    


class PatchPredictionModel(torch.nn.Module):
    def __init__(self, input_size, vit_model, num_prototypes=200, prediction_window_size=2, masking_ratio=0.8, crop_size=96, logger=None):
        super(PatchPredictionModel, self).__init__()
        self.input_size = input_size
        self.eval_spatial_resolution = input_size // 16
        self.feature_extractor = FeatureExtractor(vit_model, eval_spatial_resolution=self.eval_spatial_resolution, d_model=384)
        self.prediction_window_size = prediction_window_size
        self.CorDet = CorrespondenceDetection(window_szie=self.prediction_window_size)
        self.masking_ratio = masking_ratio
        self.crop_size = crop_size
        self.logger = logger
        self.num_prototypes = num_prototypes
        self.criterion = torch.nn.CrossEntropyLoss()
        self.key_head = torch.nn.Linear(self.feature_extractor.d_model, self.feature_extractor.d_model, bias=False)
        self.query_head = torch.nn.Linear(self.feature_extractor.d_model, self.feature_extractor.d_model, bias=False)
        self.value_head = torch.nn.Linear(self.feature_extractor.d_model, self.feature_extractor.d_model, bias=False)
        self.mlp_head = torch.nn.Sequential(
            torch.nn.Linear(self.feature_extractor.d_model, 1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, 256),
        )
        self.lc = torch.nn.Linear(self.feature_extractor.d_model, self.eval_spatial_resolution ** 2)
        self.feature_extractor.freeze_feature_extractor(["blocks.11", "blocks.10"])
        self.cross_attention_layer = CrossAttentionBlock(input_dim=self.feature_extractor.d_model, num_heads=12, dim_feedforward=2048)
        self.prototypes = torch.nn.Parameter(torch.randn(num_prototypes, 256))
    

    def normalize_prototypes(self):
        with torch.no_grad():
            w = self.prototypes.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.prototypes.copy_(w)
            
    def forward(self, imgs1, imgs2):
        bs, c, h, w = imgs1.shape
        img1_features, img1_attention = self.feature_extractor.forward_features(imgs1)
        img1_features = img1_features.reshape(bs, self.eval_spatial_resolution, self.eval_spatial_resolution, self.feature_extractor.d_model)
        img2_features, img2_attention = self.feature_extractor.forward_features(imgs2)
        img2_features = img2_features.reshape(bs, self.eval_spatial_resolution, self.eval_spatial_resolution, self.feature_extractor.d_model)
        return img1_features, img2_features
    
    def forward_sailiancy(self, img1, img2, crops):
        img1_features, img2_features = self.forward(img1, img2)
        sailiancy, revised_crop = self.CorDet(img1_features, img2_features, crops)
        return sailiancy, revised_crop
    
    def mask_features(self, features, percentage=0.8):
        """
        features: [bs, np, d_model]
        """
        bs, np, d_model = features.shape
        ## select 0.2 of the features randomly for each sample
        mask = torch.zeros(bs, np).to(features.device)
        ids = torch.randperm(np)[:int(np * percentage)]
        mask[:, ids] = 1
        mask = mask.unsqueeze(-1).repeat(1, 1, d_model)
        features = features * mask
        return features
    
    def cross_attention(self, query, key, value):
        """
        query: [bs, nq, d_model]
        key: [bs, np, d_model]
        value: [bs, np, d_model]

        return: [bs, nq, d_model]
        """
        # Parameters
        output = self.cross_attention_layer(query, key, value)
        return output

    
    def train_step(self, imgs1, imgs2):
        self.normalize_prototypes()
        bs = imgs1.shape[0]
        # crop = random_crop_mask(imgs1)
        pred_loss = 0
        clustering_loss = 0
        img1_features, img2_features = self.forward(imgs1, imgs2)
        for i in range(4):
            crop = generate_random_crop_masks(imgs1)
            sailiancy, revised_crop = self.CorDet(img1_features, img2_features, crop)
            revised_crop = F.interpolate(revised_crop.unsqueeze(1), size=(imgs1.size(-2), imgs1.size(-1)), mode='nearest').squeeze(1)
            ## find the szie of revised_crop where the value is not 0
            idxs = (revised_crop[0] != 0).nonzero()
            min_x, max_x = idxs[:, 0].min(), idxs[:, 0].max()
            min_y, max_y = idxs[:, 1].min(), idxs[:, 1].max()
            h = max_x - min_x + 1
            w = max_y - min_y + 1
            crop_mask = revised_crop > 0
            ## select the cropped area and the corresponding labels 
            cropped_area = imgs1[crop_mask.unsqueeze(1).repeat(1, 3, 1, 1)]
            cropped_labels = sailiancy
            cropped_area = cropped_area.reshape(bs, 3, h, w)
            cropped_area = torch.nn.functional.interpolate(cropped_area, size=(96, 96), mode='bilinear')
            cropped_labels = torch.nn.functional.interpolate(cropped_labels.float().unsqueeze(1), size=(96, 96), mode='nearest').squeeze(1)
            # cropped_labels = torch.arange(0, 196).reshape(14, 14).unsqueeze(0).repeat(bs, 1, 1)
            cropped_labels = cropped_labels.long().to(imgs1.device)
            masked_features2 = self.mask_features(img2_features.flatten(1, 2), self.masking_ratio)
            cropped_area_features, _ = self.feature_extractor.forward_features(cropped_area) ## size (bs, 36, d_model)
            cross_attented_features = self.cross_attention(self.query_head(cropped_area_features), self.key_head(masked_features2), self.value_head(masked_features2)) ## size (bs, 36, d_model)
            cross_attented_features = cross_attented_features.reshape(bs, 6, 6, cross_attented_features.size(-1)).permute(0, 3, 1, 2) ## size (bs, 196, 6, 6)
            resized_cross_attented_features = torch.nn.functional.interpolate(cross_attented_features, size=(96, 96), mode='bilinear').permute(0, 2, 3, 1)
            predictions = self.lc(resized_cross_attented_features) ## size (bs, 36, 196)
            predictions = predictions.permute(0, 3, 1, 2) ## size (bs, 196, 6, 6)


            cropped_area_features = self.mlp_head(cropped_area_features)
            cropped_area_features = cropped_area_features.reshape(bs, 6, 6, -1).permute(0, 3, 1, 2)
            cropped_area_features = F.interpolate(cropped_area_features, size=(96, 96), mode='bilinear').permute(0, 2, 3, 1)
            cropped_area_features = cropped_area_features.flatten(0, 2)
            normalized_crop_features = F.normalize(cropped_area_features, dim=-1)
            crop_scores = torch.einsum('bd,nd->bn', normalized_crop_features , self.prototypes)
            crop_scores = crop_scores.reshape(bs, 96, 96, self.num_prototypes).permute(0, 3, 1, 2)
            ## replace numbers in cropped_labels with the corresponding prototype idx in q

            projected_img2_features = self.mlp_head(img2_features)
            projected_img2_features = F.normalize(projected_img2_features, dim=-1)
            scores = torch.einsum('bd,nd->bn', projected_img2_features.flatten(0, -2), self.prototypes)
            q = find_optimal_assignment(scores, 0.05, 3)
            q = q.reshape(bs, self.eval_spatial_resolution, self.eval_spatial_resolution, self.num_prototypes).permute(0, 3, 1, 2)
            q = q.argmax(1)
            cropped_q_gt = q.flatten(1, 2)[torch.arange(q.size(0)).unsqueeze(1), sailiancy.flatten(1, 2)].reshape(bs, sailiancy.size(-2), sailiancy.size(-1))
            resized_crop_gt = F.interpolate(cropped_q_gt.float().unsqueeze(1), size=(96, 96), mode='nearest').squeeze(1)
            # predictions = predictions.permute(0, 3, 1, 2) ## size (bs, 196, 6, 6)
            # predictions = cross_attented_features
            # predictions = torch.nn.functional.interpolate(predictions, size=(96, 96), mode='bilinear')
            pred_loss += self.criterion(predictions, cropped_labels)
            clustering_loss += self.criterion(crop_scores / 0.1, resized_crop_gt.long())
        return pred_loss / 4, clustering_loss / 4
    

    def get_optimization_params(self):
        return [
            {"params": self.feature_extractor.parameters(), "lr": 1e-5},
            {"params": self.key_head.parameters(), "lr": 1e-4},
            {"params": self.query_head.parameters(), "lr": 1e-4},
            {"params": self.value_head.parameters(), "lr": 1e-4},
            {"params": self.lc.parameters(), "lr": 1e-4},
            {"params": self.prototypes, "lr": 1e-4},
            {"params": self.mlp_head.parameters(), "lr": 1e-4},
            {"params": self.cross_attention_layer.parameters(), "lr": 1e-4},
        ]


    def visualize(self, img1, img2, crops, epoch=0):
        ## denormalize with imagenet stats
        device = img1.device
        dn_img1 = img1 * torch.Tensor([0.225, 0.225, 0.225]).to(device).view(1, 3, 1, 1) + torch.Tensor([0.45, 0.45, 0.45]).to(device).view(1, 3, 1, 1)
        dn_img2 = img2 * torch.Tensor([0.225, 0.225, 0.225]).to(device).view(1, 3, 1, 1) + torch.Tensor([0.45, 0.45, 0.45]).to(device).view(1, 3, 1, 1)
        dn_img1 = dn_img1[0]
        dn_img2 = dn_img2[0]
        sailiancies, revised_crops = self.forward_sailiancy(img1, img2, crops)
        crop = crops[0]
        sailiancy = sailiancies[0]
        revised_crop = revised_crops[0]
        revised_sailiancy = torch.zeros((self.feature_extractor.eval_spatial_resolution, self.feature_extractor.eval_spatial_resolution))
        sailiancy_h, sailiancy_w = sailiancy.shape
        print(sailiancy.unique())
        rows = sailiancy // self.feature_extractor.eval_spatial_resolution
        cols = sailiancy % self.feature_extractor.eval_spatial_resolution
        unique_numbers = torch.unique(sailiancy)
        revised_crop[(revised_crop == 1)] = sailiancy.cpu().float().flatten()
        revised_crop = revised_crop.reshape(self.feature_extractor.eval_spatial_resolution, self.feature_extractor.eval_spatial_resolution)
        for i in range(sailiancy_h):
            for j in range(sailiancy_w):
                revised_sailiancy[rows[i, j], cols[i, j]] = sailiancy[i, j]
        for i, number in enumerate(unique_numbers.cpu()):
            revised_sailiancy[revised_sailiancy == number] = i + 1
            revised_crop[revised_crop == number] = i + 1

        revised_crop = torch.nn.functional.interpolate(revised_crop.unsqueeze(0).unsqueeze(0), size=(dn_img1.size(-2), dn_img1.size(-1)), mode='nearest').squeeze(0).squeeze(0)
        plt.imshow(dn_img1.permute(1, 2, 0).detach().cpu().numpy())
        plt.imshow(crop.cpu().numpy(), alpha=0.5)
        plt.savefig(f"Temp/{epoch}_overlayed_img1.png")
        overlaied_img2 = overlay(dn_img2.permute(1, 2, 0).detach().cpu().numpy(), crop.cpu().numpy())
        overlaied_img2 = torch.from_numpy(overlaied_img2).permute(2, 0, 1)
        # wandb.log({"overlaied_img2": wandb.Image(overlaied_img2)})
        plt.imshow(dn_img1.permute(1, 2, 0).detach().cpu().numpy())
        plt.imshow(revised_crop.cpu().numpy(), alpha=0.5, cmap=cmap)
        plt.savefig(f"Temp/{epoch}_revised_crop_img1.png")
        plt.imshow(dn_img2.permute(1, 2, 0).detach().cpu().numpy())
        resized_revised_sailiancy = torch.nn.functional.interpolate(revised_sailiancy.unsqueeze(0).unsqueeze(0), size=(dn_img2.size(-2), dn_img2.size(-1)), mode='nearest').squeeze(0).squeeze(0)
        plt.imshow(resized_revised_sailiancy.detach().cpu().numpy(), alpha=0.5, cmap=cmap)
        plt.savefig(f"Temp/{epoch}_overlayed_img2.png")
        # overlaied_img1 = overlay(dn_img1.permute(1, 2, 0).detach().cpu().numpy(), sailiancy.detach().cpu().numpy())
        # overlaied_img1 = torch.from_numpy(overlaied_img1).permute(2, 0, 1)
        # wandb.log({"overlaied_img1": wandb.Image(overlaied_img1)})


    def validate_step(self, img):
        self.feature_extractor.eval()
        with torch.no_grad():
            spatial_features, _ = self.feature_extractor.forward_features(img)  # (B, np, dim)
        return spatial_features
    

    def validate_step1(self, img):
        self.feature_extractor.eval()
        with torch.no_grad():
            bs = img.shape[0]
            crop = random_crop_mask(img)
            img1_features, img2_features = self.forward(img, img)
            sailiancy, revised_crop = self.CorDet(img1_features, img2_features, crop)
            revised_crop = F.interpolate(revised_crop.unsqueeze(1), size=(img.size(-2), img.size(-1)), mode='nearest').squeeze(1)
            ## find the szie of revised_crop where the value is not 0
            idxs = (revised_crop != 0).nonzero()
            min_x, max_x = idxs[:, 1].min(), idxs[:, 1].max()
            min_y, max_y = idxs[:, 2].min(), idxs[:, 2].max()
            h = max_x - min_x + 1
            w = max_y - min_y + 1
            crop_mask = revised_crop > 0
            ## select the cropped area and the corresponding labels 
            cropped_area = img[crop_mask.unsqueeze(1).repeat(1, 3, 1, 1)]
            cropped_labels = sailiancy
            cropped_area = cropped_area.reshape(bs, 3, h, w)
            cropped_area = torch.nn.functional.interpolate(cropped_area, size=(96, 96), mode='bilinear')
            cropped_labels = torch.nn.functional.interpolate(cropped_labels.float().unsqueeze(1), size=(96, 96), mode='nearest').squeeze(1)
            # cropped_labels = torch.arange(0, 196).reshape(14, 14).unsqueeze(0).repeat(bs, 1, 1)
            cropped_labels = cropped_labels.long().to(img.device)
            masked_features2 = self.mask_features(img1_features.flatten(1, 2), self.masking_ratio)
            cropped_area_features, _ = self.feature_extractor.forward_features(cropped_area) ## size (bs, 36, d_model)
            cross_attented_features = self.cross_attention(self.query_head(cropped_area_features), self.key_head(masked_features2), self.value_head(masked_features2)) ## size (bs, 36, d_model)
            cross_attented_features = cross_attented_features.reshape(bs, 6, 6, cross_attented_features.size(-1)).permute(0, 2, 3, 1) ## size (bs, 196, 6, 6)
            # resized_cross_attented_features = torch.nn.functional.interpolate(cross_attented_features, size=(96, 96), mode='bilinear').permute(0, 2, 3, 1)
            predictions = self.mlp_head(cross_attented_features) ## size (bs, 36, 196)
            predictions = predictions.permute(0, 3, 1, 2) ## size (bs, 196, 6, 6)
            loss = self.criterion(predictions, cropped_labels)
            return loss

    def save(self, path):
        torch.save(self.state_dict(), path)


        

class PatchPredictionTrainer():
    def __init__(self, dataloader, test_dataloader, patch_prediction_model, num_epochs, device, logger):
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        self.patch_prediction_model = patch_prediction_model
        self.device = device
        self.patch_prediction_model = self.patch_prediction_model.to(self.device)
        self.optimizer = None
        self.num_epochs = num_epochs
        self.logger = logger
        self.logger.watch(patch_prediction_model, log="all", log_freq=10)
    
    def visualize(self):
        for i, batch in enumerate(self.dataloader):
            datum, annotations = batch
            annotations = annotations.squeeze(1)
            datum = datum.squeeze(1)
            imgs1, imgs2 = datum[:, 0], datum[:, 1]
            imgs1, imgs2 = imgs1.to(self.device), imgs2.to(self.device)
            crop = random_crop_mask(imgs1)
            self.patch_prediction_model.visualize(imgs1, imgs2, crop, i)
    
    def setup_optimizer(self, optimization_config):
        model_params = self.patch_prediction_model.get_optimization_params()
        init_lr = optimization_config['init_lr']
        peak_lr = optimization_config['peak_lr']
        decay_half_life = optimization_config['decay_half_life']
        warmup_steps = optimization_config['warmup_steps']
        grad_norm_clip = optimization_config['grad_norm_clip']
        init_weight_decay = optimization_config['init_weight_decay']
        peak_weight_decay = optimization_config['peak_weight_decay']
        ## read the first batch from dataloader to get the number of iterations
        num_itr = len(self.dataloader)
        max_itr = self.num_epochs * num_itr
        self.optimizer = PatchCorrespondenceOptimizer(model_params, init_lr, peak_lr, decay_half_life, warmup_steps, grad_norm_clip, init_weight_decay, peak_weight_decay, max_itr)
        self.optimizer.setup_optimizer()
        self.optimizer.setup_scheduler()
    

    def train_one_epoch(self):
        self.patch_prediction_model.train()
        epoch_loss = 0
        before_loading_time = time.time()
        for i, batch in enumerate(self.dataloader):
            after_loading_time = time.time()
            print("Loading Time: {}".format(after_loading_time - before_loading_time))
            datum, annotations = batch
            annotations = annotations.squeeze(1)
            datum = datum.squeeze(1)
            imgs1, imgs2 = datum[:, 0], datum[:, 1]
            # imgs1, imgs2 = datum, datum
            imgs1, imgs2 = imgs1.to(self.device), imgs2.to(self.device)
            loss, clustering_loss = self.patch_prediction_model.train_step(imgs1, imgs2)
            total_loss = 0 * loss + clustering_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.optimizer.update_lr()
            epoch_loss += total_loss.item()
            print("Iteration: {} Loss: {}".format(i, total_loss.item()))
            self.logger.log({"loss": loss.item()})
            self.logger.log({"clustering_loss": clustering_loss.item()})
            lr = self.optimizer.get_lr()
            self.logger.log({"lr": lr})
            before_loading_time = time.time()
        epoch_loss /= (i + 1)
        if epoch_loss < 2.5:
            self.patch_prediction_model.save(f"Temp/model_{epoch_loss}.pth")
        print("Epoch Loss: {}".format(epoch_loss))
    
    def train(self):
        for epoch in range(self.num_epochs):
            print("Epoch: {}".format(epoch))
            self.train_one_epoch()
            if epoch % 4 == 0:
                self.validate(epoch)
            # self.validate(epoch)
            # self.patch_prediction_model.save_model(epoch)
            # self.validate(epoch)
    
    def validate(self, epoch, val_spatial_resolution=56):
        self.patch_prediction_model.eval()
        with torch.no_grad():
            metric = PredsmIoU(21, 21)
            # spatial_feature_dim = self.model.get_dino_feature_spatial_dim()
            spatial_feature_dim = 50
            clustering_method = PerDatasetClustering(spatial_feature_dim, 21)
            feature_spatial_resolution = self.patch_prediction_model.feature_extractor.eval_spatial_resolution
            feature_group = []
            targets = []
            for i, (x, y) in enumerate(self.test_dataloader):
                target = (y * 255).long()
                img = x.to(self.device)
                spatial_features = self.patch_prediction_model.validate_step(img)  # (B, np, dim)
                resized_target =  F.interpolate(target.float(), size=(val_spatial_resolution, val_spatial_resolution), mode="nearest").long()
                targets.append(resized_target)
                feature_group.append(spatial_features)
            eval_features = torch.cat(feature_group, dim=0)
            eval_targets = torch.cat(targets, dim=0)
            B, np, dim = eval_features.shape
            eval_features = eval_features.reshape(eval_features.shape[0], feature_spatial_resolution, feature_spatial_resolution, dim)
            eval_features = eval_features.permute(0, 3, 1, 2).contiguous()
            eval_features = F.interpolate(eval_features, size=(val_spatial_resolution, val_spatial_resolution), mode="bilinear")
            eval_features = eval_features.reshape(B, dim, -1).permute(0, 2, 1)
            eval_features = eval_features.detach().cpu().unsqueeze(1)
            cluster_maps = clustering_method.cluster(eval_features)
            cluster_maps = cluster_maps.reshape(B, val_spatial_resolution, val_spatial_resolution).unsqueeze(1)
            valid_idx = eval_targets != 255
            valid_target = eval_targets[valid_idx]
            valid_cluster_maps = cluster_maps[valid_idx]
            metric.update(valid_target, valid_cluster_maps)
            jac, tp, fp, fn, reordered_preds, matched_bg_clusters = metric.compute(is_global_zero=True)
            self.logger.log({"val_k=gt_miou": jac})
            # print(f"Epoch : {epoch}, eval finished, miou: {jac}")
    

    def validate1(self, epoch, val_spatial_resolution=56):
        self.patch_prediction_model.eval()
        losses = []
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_dataloader):
                target = (y * 255).long()
                img = x.to(self.device)
                loss = self.patch_prediction_model.validate_step1(img)  # (B, np, dim)
                resized_target =  F.interpolate(target.float(), size=(val_spatial_resolution, val_spatial_resolution), mode="nearest").long()
                losses.append(loss.item())
            avg_loss = sum(losses) / len(losses)
            self.logger.log({"val_loss": avg_loss})



    def train_lc(self, lc_train_dataloader, lc_val_dataloader):
        best_miou = 0
        for epoch in range(self.num_epochs):
            print("Epoch: {}".format(epoch))
            if epoch % 60 == 64:
                val_miou = self.lc_validation(lc_train_dataloader, lc_val_dataloader, self.device)
                if val_miou > best_miou:
                    best_miou = val_miou
                    self.patch_prediction_model.save(f"Temp/model_{epoch}_{best_miou}.pth")
            self.train_one_epoch()
            # self.validate(epoch)
            # self.patch_prediction_model.save_model(epoch)
            # self.validate(epoch)

    def lc_validation(self, train_dataloader, val_dataloader, device):
        self.patch_prediction_model.eval()
        model = self.patch_prediction_model.feature_extractor
        lc_module = LinearFinetuneModule(model, train_dataloader, val_dataloader, device)
        final_miou = lc_module.linear_segmentation_validation()
        self.logger.log({"lc_val_miou": final_miou})
        return final_miou




def run(args):
    device = args.device
    ucf101_path = args.ucf101_path
    clip_durations = args.clip_durations
    batch_size = args.batch_size
    num_workers = args.num_workers
    input_size = args.input_size
    num_epochs = args.num_epochs
    masking_ratio = args.masking_ratio
    crop_size = args.crop_size
    crop_scale = args.crop_scale_tupple

    config = vars(args)
    ## make a string of today's date
    today = date.today()
    d1 = today.strftime("%d_%m_%Y")
    logger = wandb.init(project=project_name, group=d1, job_type='debug_clustering_ytvos', config=config)
    rand_color_jitter = video_transformations.RandomApply([video_transformations.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.8)
    data_transform_list = [rand_color_jitter, video_transformations.RandomGrayscale(), video_transformations.RandomGaussianBlur()]
    data_transform = video_transformations.Compose(data_transform_list)
    video_transform_list = [video_transformations.RandomResizedCrop((224, 224), scale=crop_scale), video_transformations.ClipToTensor(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])] #video_transformations.RandomResizedCrop((224, 224))
    video_transform = video_transformations.Compose(video_transform_list)
    num_clips = 1
    num_clip_frames = 2
    regular_step = 1
    transformations_dict = {"data_transforms": data_transform, "target_transforms": None, "shared_transforms": video_transform}
    prefix = "/ssdstore/ssalehi/dataset"
    data_path = os.path.join(prefix, "train1/JPEGImages/")
    annotation_path = os.path.join(prefix, "train1/Annotations/")
    meta_file_path = os.path.join(prefix, "train1/meta.json")
    path_dict = {"class_directory": data_path, "annotation_directory": annotation_path, "meta_file_path": meta_file_path}
    sampling_mode = SamplingMode.DENSE
    video_data_module = VideoDataModule("ytvos", path_dict, num_clips, num_clip_frames, sampling_mode, regular_step, batch_size, num_workers)
    video_data_module.setup(transformations_dict)
    data_loader = video_data_module.get_data_loader()

    vit_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    patch_prediction_model = PatchPredictionModel(224, vit_model, masking_ratio=masking_ratio, crop_size=crop_size, logger=logger)
    optimization_config = {
        'init_lr': 1e-4,
        'peak_lr': 1e-3,
        'decay_half_life': 0,
        'warmup_steps': 0,
        'grad_norm_clip': 1.0,
        'init_weight_decay': 1e-2,
        'peak_weight_decay': 1e-2
    }
    image_val_transform = trn.Compose([trn.Resize((input_size, input_size)), trn.ToTensor(), trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])])
    shared_val_transform = Compose([
        Resize(size=(input_size, input_size)),
    ])
    val_transforms = {"img": image_val_transform, "target": None , "shared": shared_val_transform}
    dataset = PascalVOCDataModule(batch_size=batch_size, train_transform=val_transforms, val_transform=val_transforms, test_transform=val_transforms, num_workers=num_workers)
    dataset.setup()
    test_dataloader = dataset.get_test_dataloader()
    patch_prediction_trainer = PatchPredictionTrainer(data_loader, test_dataloader, patch_prediction_model, num_epochs, device, logger)
    patch_prediction_trainer.setup_optimizer(optimization_config)
    patch_prediction_trainer.train()

    # patch_prediction_trainer.visualize()


            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:3")
    parser.add_argument('--ucf101_path', type=str, default="/ssdstore/ssalehi/ucf101/data/UCF101")
    parser.add_argument('--clip_durations', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--num_epochs', type=int, default=800)
    parser.add_argument('--crop_size', type=int, default=64)
    parser.add_argument('--crop_scale_tupple', type=tuple, default=(0.3, 1))
    parser.add_argument('--masking_ratio', type=float, default=1)
    parser.add_argument('--same_frame_query_ref', type=bool, default=False)
    parser.add_argument("--explaination", type=str, default="clustering, every other thing is the same; except the crop and reference are not of the same frame. and num_crops =4")
    args = parser.parse_args()
    run(args)



        
