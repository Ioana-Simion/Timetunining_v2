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

class LinearFinetune(torch.nn.Module):
    def __init__(self, model, train_epoch: int, num_classes: int, lr: float, input_size: int, spatial_res: int, val_iters: int,
                 drop_at: int, arch: str, head_type: str = None, decay_rate: float = 0.1, ignore_index: int = 255, device=None):
        super().__init__()
        # Init Model
        # if 'vit' in arch:
            # self.model = create_model(f'{arch}_patch{patch_size}_224', pretrained=False)
        self.model = model
        self.model_embed_dim = self.model.get_mae_feature_spatial_dim()
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
            tokens = self.model.forward_encoder(imgs)
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
            tokens = self.model.forward_encoder(imgs)
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




def unsupervised_segmentation_validattion(self, epoch, val_spatial_resolution=56):
    self.model.eval()
    with torch.no_grad():
        metric = PredsmIoU(21, 21)
        # spatial_feature_dim = self.model.get_dino_feature_spatial_dim()
        spatial_feature_dim = 50
        clustering_method = PerDatasetClustering(spatial_feature_dim, 21)
        feature_spatial_resolution = self.model.get_dino_feature_spatial_resolution()
        mae_feature_group = []
        targets = []
        for i, (x, y) in enumerate(self.test_dataloader):
            target = (y * 255).long()
            img = x.to(self.device)
            mae_spatial_features = self.model.forward_encoder(img)  # (B, np, dim)
            resized_target =  F.interpolate(target.float(), size=(val_spatial_resolution, val_spatial_resolution), mode="nearest").long()
            targets.append(resized_target)
            mae_feature_group.append(mae_spatial_features)
        eval_features = torch.cat(mae_feature_group, dim=0)
        eval_targets = torch.cat(targets, dim=0)
        B, np, dim = eval_features.shape
        eval_features = eval_features.reshape(eval_features.shape[0], feature_spatial_resolution, feature_spatial_resolution, dim)
        eval_features = eval_features.permute(0, 3, 1, 2)
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
        print(f"eval finished, miou: {jac}")