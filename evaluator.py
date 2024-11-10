import copy
import torch
import numpy as np
from torchmetrics import Metric
from typing import Optional, List, Tuple, Dict
import os
import time
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from data_loader import PascalVOCDataModule
import wandb
from timm.models import create_model
from models import FCNHead
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from eval_metrics import PredsmIoU
from clustering import PerDatasetClustering
import torch.nn.functional as F

import torch
from einops import einsum
from tqdm import tqdm

class LinearFinetune(torch.nn.Module):
    def __init__(self, model, num_classes: int, lr: float, input_size: int, spatial_res: int, val_iters: int,
                 drop_at: int, arch: str, head_type: str = None, decay_rate: float = 0.1, ignore_index: int = 255, device=None):
        super().__init__()
        # Init Model
        # if 'vit' in arch:
            # self.model = create_model(f'{arch}_patch{patch_size}_224', pretrained=False)
        self.model = model
        self.model_embed_dim = self.model.d_model
        if head_type == "fcn":
            self.finetune_head = FCNHead(
                in_channels=self.model_embed_dim,
                channels=512,
                num_convs=2,
                concat_input=True,
                dropout_ratio=0.1,
                num_classes=num_classes,
            )
        else:
            self.finetune_head = torch.nn.Conv2d(self.model_embed_dim, num_classes, 1)
        
        self.finetune_head = self.finetune_head.to(device)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.miou_metric = PredsmIoU(num_classes, num_classes)
        self.num_classes = num_classes
        self.lr = lr
        self.val_iters = val_iters
        self.input_size = input_size
        self.spatial_res = spatial_res
        self.drop_at = drop_at
        self.arch = arch
        self.ignore_index = ignore_index
        self.decay_rate = decay_rate
        self.train_mask_size = 100
        self.val_mask_size = 100
        self.device = device
        self.optimizer, self.scheduler = self.configure_optimizers()

    def on_after_backward(self):
        # Freeze all layers of backbone
        for param in self.model.parameters():
            param.requires_grad = False

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.finetune_head.parameters(), weight_decay=0.0001,
                                    momentum=0.9, lr=self.lr)
        scheduler = StepLR(optimizer, gamma=self.decay_rate, step_size=self.drop_at)
        return optimizer, scheduler

    def train_step(self, batch):
        self.finetune_head.train()
        imgs, masks = batch
        imgs = imgs.to(self.device)
        masks = masks.to(self.device)
        bs = imgs.size(0)
        res = imgs.size(3)
        assert res == self.input_size
        self.model.eval()

        with torch.no_grad():
            tokens, _ = self.model.forward_features(imgs)
            if 'vit' in self.arch:
                tokens = tokens.reshape(bs, self.spatial_res, self.spatial_res, self.model_embed_dim).permute(0, 3, 1, 2)
            tokens = nn.functional.interpolate(tokens, size=(self.train_mask_size, self.train_mask_size),
                                               mode='bilinear')
        mask_preds = self.finetune_head(tokens)

        masks *= 255
        if self.train_mask_size != self.input_size:
            with torch.no_grad():
                masks = nn.functional.interpolate(masks, size=(self.train_mask_size, self.train_mask_size),
                                                  mode='nearest')

        loss = self.criterion(mask_preds, masks.long().squeeze())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss

    def validation_step(self, batch):
        self.finetune_head.eval()
        with torch.no_grad():
            imgs, masks = batch
            bs = imgs.size(0)
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)
            tokens, _ = self.model.forward_features(imgs)
            tokens = tokens.reshape(bs, self.spatial_res, self.spatial_res, self.model_embed_dim).permute(0, 3, 1, 2)
            tokens = nn.functional.interpolate(tokens, size=(self.val_mask_size, self.val_mask_size),
                                                mode='bilinear')
            mask_preds = self.finetune_head(tokens)

            # downsample masks and preds
            gt = masks * 255
            gt = nn.functional.interpolate(gt, size=(self.val_mask_size, self.val_mask_size), mode='nearest')
            valid = (gt != self.ignore_index) # mask to remove object boundary class
            mask_preds = torch.argmax(mask_preds, dim=1).unsqueeze(1)

            # update metric
            self.miou_metric.update(gt[valid], mask_preds[valid])

    def validation_epoch_end(self):
        miou = self.miou_metric.compute(True, many_to_one=False, linear_probe=True)[0]
        self.miou_metric.reset()
        return miou
    



class LinearFinetuneModule():
    def __init__(self, model, train_dataloader, val_dataloader, device, spatial_resolution=14, train_epoch=20, drop_at=20):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.train_epoch = train_epoch
        self.spatial_resolution = spatial_resolution
        self.drop_at = drop_at
        total_iters = len(self.train_dataloader) * self.drop_at
        cloned_model = copy.deepcopy(self.model)
        self.linear_evaluator = LinearFinetune(cloned_model,  num_classes=21, lr=0.01, input_size=224, spatial_res=self.spatial_resolution, val_iters=20,
                    drop_at=total_iters, arch="vit_small", head_type="lc", device=self.device)

    def linear_segmentation_validation(self):

        ## keep a dictionary of a few parameters of the model and later check if they are changed
        ########################################################
        # dict = {}
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         dict[name] = param.data.clone()

        ########################################################
        
        ########################################################
        ## check if the parameters are changed
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         assert torch.equal(param.data, dict[name])

        ########################################################
        

        print("==============================================================")
        final_miou = 0
        for j in range(self.train_epoch):
            for i, (x, y) in enumerate(self.train_dataloader):
                loss = self.linear_evaluator.train_step((x, y))
                print('linear_eval_loss', loss)
        for i, (x, y) in enumerate(self.val_dataloader):
            self.linear_evaluator.validation_step((x, y))
        miou = self.linear_evaluator.validation_epoch_end()
        final_miou = miou
        print('miou_val', round(miou, 6))
        return final_miou

def argmax_2d(x, max_value=True):
    h, w = x.shape[-2:]
    x = torch.flatten(x, start_dim=-2)
    if max_value:
        flat_indices = x.argmax(dim=-1)
    else:
        flat_indices = x.argmin(dim=-1)

    min_row = flat_indices // w
    min_col = flat_indices % w
    xy_indices = torch.stack((min_col, min_row), dim=-1)
    return xy_indices

class KeypointMatchingModule():
    def __init__(self, model, dataset, device, threshold=0.10):
        """
        Initialize the Keypoint Matching Evaluation module.
        Args:
            model: The feature extractor model to evaluate.
            dataset: The SPair dataset for keypoint matching evaluation.
            device: Device to perform computations on ('cuda' or 'cpu').
            threshold: Distance threshold for keypoint matching success.
        """
        self.model = model
        self.dataset = dataset
        self.device = device
        self.threshold = threshold
        self.model.to(self.device)
        self.model.eval()

    def compute_errors(self, instance):
        # Compute keypoint matching -- probe 3d logic
        print(f"Instance has {len(instance)} elements:", instance)
        # Adjust unpacking based on the actual return structure of instance
        img_i, mask_i, kps_i, img_j, mask_j, kps_j, thresh_scale, *rest = instance
        #img_i, mask_i, kps_i, img_j, mask_j, kps_j, _ = instance
        mask_i = torch.tensor(np.array(mask_i, dtype=float))
        mask_j = torch.tensor(np.array(mask_j, dtype=float))

        images = torch.stack((img_i, img_j)).to(self.device)
        masks = torch.stack((mask_i, mask_j)).to(self.device)
        masks = F.avg_pool2d(masks.float(), 16)
        masks = masks > 4 / (16**2)

        #feats = self.model(images)
        #feats = self.model.forward_features(images)
        feats, _ = self.model.feature_extractor.forward_features(images)
        batch_size, num_patches, channels = feats.shape
        grid_size = int(num_patches ** 0.5)  # This should be 16 if patches are in a 16x16 grid

        if grid_size ** 2 == num_patches:
            feats = feats.view(batch_size, grid_size, grid_size, channels).permute(0, 3, 1, 2)
            print("Reshaped feats shape for grid_sample:", feats.shape)  # Should print [2, 384, 16, 16]
        else:
            print("Unexpected number of patches, could not reshape.")

        feats = F.normalize(feats, p=2, dim=1)
        feats_i = feats[0]
        feats_j = feats[1]
 
        kps_i = kps_i.float() / images.shape[-1]
        kps_j = kps_j.float() / images.shape[-1]
        
        kps_i_ndc = (kps_i[:, :2] * 2 - 1).unsqueeze(0).unsqueeze(0).to(self.device)
        print("kps_i_ndc shape:", kps_i_ndc.shape)  
        kp_i_F = F.grid_sample(feats_i.unsqueeze(0), kps_i_ndc, mode="bilinear", align_corners=True)[0, :, 0].t()

        heatmaps = einsum(kp_i_F, feats_j, "k f, f h w -> k h w")
        pred_kp = argmax_2d(heatmaps, max_value=True).float().cpu() / feats.shape[-1]

        errors = (pred_kp[:, None, :] - kps_j[None, :, :2]).norm(p=2, dim=-1)
        errors /= self.threshold

        valid_kps = (kps_i[:, 2] * kps_j[:, 2]) == 1
        in_both = valid_kps.diagonal()
        errors[~valid_kps] = 1e3

        return errors.diagonal()[in_both], errors[in_both].min(dim=1)[0]

    def evaluate(self):
        """
        Evaluate the model on keypoint matching over the dataset.
        Returns:
            recall: Keypoint matching recall score.
        """
        all_errors = []
        for i in tqdm(range(len(self.dataset)), ncols=60):
            errors_same, errors_nn = self.compute_errors(self.dataset[i])
            all_errors.extend(errors_same)
        
        recall = (torch.tensor(all_errors) < self.threshold).float().mean().item() * 100.0
        return recall
