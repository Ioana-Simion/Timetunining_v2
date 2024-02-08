import argparse
from datetime import date
import time
import numpy as np
import torch.nn.functional as F
from clustering import PerDatasetClustering
from data_loader import CocoDataModule, ImangeNet_100_Handler, PascalVOCDataModule
from dino_vision_transformer import vit_base, VisionTransformer
from eval_metrics import PredsmIoU
from evaluator import LinearFinetuneModule
from my_utils import find_optimal_assignment, overlay_video_cmap, process_attentions, sinkhorn
import wandb
import torchvision.transforms as trn
from image_transformations import RandomResizedCrop, RandomHorizontalFlip, Compose, Resize
from matplotlib import pyplot as plt
import timm
import torch
from functools import partial
import torch.nn as nn
from PIL import ImageFilter, Image
from optimizer import TimeTv2Optimizer
import random
from models import SlotAttention



class GaussianBlur:
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709 following
    https://github.com/facebookresearch/swav/blob/5e073db0cc69dea22aa75e92bfdd75011e888f28/src/multicropdataset.py#L64
    """
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x: Image):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def generate_mask_with_attention_guidance(cluster_maps, mask_ratio=0.5):
    """ 
    cluster_maps: [batch_size, spatial_res * spatial_res]
    generate a mask matrix that mask patches of each cluster with the specified mask ratio
    """
    batch_size = cluster_maps.size(0)
    spatial_res = cluster_maps.size(1)
    num_clusters = cluster_maps.max() + 1
    mask = torch.zeros(batch_size, spatial_res)
    for i in range(batch_size):
        for j in range(num_clusters):
            cluster_map = cluster_maps[i]
            cluster_map = cluster_map == j
            num_patches = cluster_map.sum()
            num_mask_patches = int(num_patches * mask_ratio)
            mask_patches = torch.randperm(num_patches)[:num_mask_patches]
            mask_patches = cluster_map.nonzero()[mask_patches]
            mask_patches = mask_patches[:, 0]
            mask[i, mask_patches] = 1
    
    boolean_mask = mask == 1
    return boolean_mask, mask_patches

def generate_random_mask(data, mask_ratio=0.5):
    masks = torch.zeros_like(data)
    batch_size = data.size(0)
    num_pathes = data.size(1)
    num_mask_patches = int(num_pathes * mask_ratio)
    ids_to_keep = []
    for i in range(batch_size):
        mask_patches = torch.randperm(num_pathes)[:num_mask_patches]
        unmask_patches = torch.randperm(num_pathes)[num_mask_patches:]
        masks[i, mask_patches] = 1
        sorted_unmask_patches = unmask_patches.sort()[0]
        ids_to_keep.append(sorted_unmask_patches)
    ids_to_keep = torch.stack(ids_to_keep)
    boolean_mask = masks == 1
    return boolean_mask, ids_to_keep




class MyVisionTransformer(VisionTransformer):

    def __init__(self, depth=12, spatial_resolution=14, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.spatial_resolution = spatial_resolution


    def get_intermediate_layer_feats(self, imgs, feat="k", layer_num=-1):
        bs, c, h, w = imgs.shape
        imgs = imgs.reshape(bs, c, h, w)
        ## hook to get the intermediate layers
        feat_out = {}
        def hook_fn_forward_qkv(module, input, output):
            feat_out["qkv"] = output
        self._modules["blocks"][layer_num]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)
        attentions = self.get_last_selfattention(imgs)
        # Scaling factor
        average_cls_attention = torch.mean(attentions[:, :, 0, 1:], dim=1)
        temp_mins, temp_maxs = average_cls_attention.min(dim=1)[0], average_cls_attention.max(dim=1)[0]
        normalized_cls_attention = (average_cls_attention - temp_mins[:, None]) / (temp_maxs[:, None] - temp_mins[:, None])
        # cls_attentions = process_attentions(attentions[:, :, 0, 1:], self.spatial_resolution)  
        # Dimensions
        nb_im = attentions.shape[0]  # Batch size
        nh = attentions.shape[1]  # Number of heads
        nb_tokens = attentions.shape[2]  # Number of tokens
        qkv = (
            feat_out["qkv"]
            .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
        q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
        v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

        if feat == "k":
            feats = k[:, 1:, :]
        elif feat == "q":
            feats = q[:, 1:, :]
        elif feat == "v":
            feats = v[:, 1:, :]
        return feats, normalized_cls_attention

    def forward_features(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 1:]

    

    def masking_forward_features(self, x, mask_ratio=0.5):
        """
        x: [batch_size, c, h, w]
        mask: [batch_size, spatial_resolution * spatial_resolution] 
        mask is a binary matrix that indicates which patches are masked
        """
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding
        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)
        
        if mask_ratio != 0:
            mask, ids_to_keep = generate_random_mask(x[:, 1:], mask_ratio=mask_ratio)
        mask = mask.to(x.device)
        unmasked = ~mask
        cls_tokens = x[:, 0]
        x = x[:, 1:]
        x = x[unmasked].reshape(B, -1, x.size(-1))
        x = torch.cat((cls_tokens.unsqueeze(1), x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x[:, 1:], ids_to_keep

def test_my_vision_transformer(model, input):
    _, attn = model.get_last_selfattention(input)


def visualized_attention_map(model, input, head_num=0, threshold=0.1):
    eval_spatial_res = model.spatial_resolution
    attn = model.get_last_selfattention(input)
    cls_attn = attn[:, :, 0, 1:]
    if head_num == -1:
        attn = process_attentions(cls_attn, spatial_res=eval_spatial_res, threshold=threshold)
    else:
        attn = cls_attn[:, head_num]
        attn = attn.view(-1, 1, eval_spatial_res, eval_spatial_res)
        attn_threshold = attn > threshold
        attn[attn_threshold] = 1
        attn[~attn_threshold] = 0
    return attn


def log_attention_maps_on_pascal(model, data, logger, num_head=-1, input_size=224):
    logger.log({"image": wandb.Image(denormalized_data)})
    if num_head == -1:
        attn = visualized_attention_map(model, data, head_num=-1)
        resized_attn = F.interpolate(attn, size=(input_size, input_size), mode="nearest")
        denormalized_data = data * torch.tensor([0.229, 0.224, 0.255]).view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        denormalized_data = (denormalized_data * 255).type(torch.uint8)
        _, overlayed_data = overlay_video_cmap(resized_attn.squeeze(1), denormalized_data)
        logger.log({"overlayed_images": wandb.Image(overlayed_data)})
    else:
        for i in range(num_head):
            attn = visualized_attention_map(model, data, head_num=i)
            resized_attn = F.interpolate(attn, size=(input_size, input_size), mode="nearest")
            _, overlayed_data = overlay_video_cmap(resized_attn.squeeze(1), denormalized_data)
            logger.log({f"overlayed_images_head_{i}": wandb.Image(overlayed_data)})
    logger.finish()



def log_attention_heads_on_pascal(device, model, logger, data):
    input_size = data.size(-1)
    denormalized_data = data * torch.tensor([0.229, 0.224, 0.255]).view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    denormalized_data = (denormalized_data * 255).type(torch.uint8)
    data = data.to(device)
    eval_spatial_res = model.spatial_resolution
    attn = model.get_last_selfattention(data)
    cls_attn_prev = attn[:, :, 0, 1:]
    cls_attn = cls_attn_prev.reshape(-1, cls_attn_prev.size(1), eval_spatial_res, eval_spatial_res)
    cls_attn = F.interpolate(cls_attn, size=(input_size, input_size), mode="nearest")
    for j, d in enumerate(denormalized_data):
        # Create a subplot for each attention head and the original image
        fig, axs = plt.subplots(4, 4, figsize=(12, 12))  # You can adjust the figsize as needed

        # Plot the original image
        axs[0, 0].imshow(d.permute(1, 2, 0))
        axs[0, 0].axis('off')  # Optionally turn off axis

        # Plot each attention map
        for i in range(1, 13):
            axs[i // 4, i % 4].imshow(cls_attn[j, i - 1].cpu().detach().numpy())
            axs[i // 4, i % 4].axis('off')  # Optionally turn off axis

        # Remove empty subplots
        for ax in axs.flat:
            if not ax.images:
                fig.delaxes(ax)

        fig.tight_layout(pad=1.0)  # Adjust layout

        ## create the histogram of the attention heads
        fig1, axs1 = plt.subplots(4, 3, figsize=(12, 12))
        for i in range(0, 12):
            axs1[i // 3, i % 3].hist(cls_attn_prev[j, i - 1].cpu().detach().numpy().reshape(-1), bins=10)
            axs1[i // 3, i % 3].set_title(f"Attention Map {i}")
            axs1[i // 3, i % 3].set_xlim(0, 1)
        
        
        fig1.tight_layout(pad=1.0)


        # Create wandb Image
        wandb_image = wandb.Image(fig, caption=f"Attention Map {j}")
        wandb_image1 = wandb.Image(fig1, caption=f"Attention Map {j}_hist")

        # log image and image1 together
        logger.log({"attention_map": wandb_image, "attention_map_hist": wandb_image1})

        plt.close(fig)  # Close the figure to free memory
        plt.close(fig1)
            

        ## use matplotlib to visualize the attention ma


def create_prototype_clustermap(device, model, logger, data_loader):
    data, target = next(iter(data_loader))
    input_size = data.size(-1)
    denormalized_data = data * torch.tensor([0.229, 0.224, 0.255]).view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    denormalized_data = (denormalized_data * 255).type(torch.uint8)
    data = data.to(device)
    eval_spatial_res = model.spatial_resolution
    attn = model.get_last_selfattention(data)
    cls_attn = attn[:, :, 0, 1:]
    value_attention = attn[:, :, 1:, 1:]
    head_num = value_attention.size(1)
    ## normalize the attention heads between 0 and 1
    # cls_attn = cls_attn / cls_attn.sum(dim=-1, keepdim=True)

    queries, _ = model.get_intermediate_layer_feats(data, feat="q", layer_num=-1)
    keys, _ = model.get_intermediate_layer_feats(data, feat="k", layer_num=-1)
    values, _ = model.get_intermediate_layer_feats(data, feat="v", layer_num=-1)
    # queries = queries / queries.norm(dim=-1, keepdim=True)
    last_features = model.forward_features(data)
    prototypes = torch.einsum("bhp,bpd->bhd", cls_attn, last_features)
    ## normalize the prototypes
    prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)
    cluster_maps = []
    values, _ = model.get_intermediate_layer_feats(data, feat="v", layer_num=-1)
    values = values.reshape(values.size(0), values.size(1), head_num, values.size(2) // head_num).permute(0, 2, 1, 3)
    attention_v =  torch.einsum("bhpcd,bhppc->bhpd", values.unsqueeze(3), value_attention.unsqueeze(-1))
    attention_v = attention_v.permute(0, 2, 1, 3)
    attention_v = attention_v.flatten(2, -1)
    # features = model.forward_features(data)
    # features = features[:, 1:, :]
    features = last_features
    features = features / features.norm(dim=-1, keepdim=True)
    for i, prototype in enumerate(prototypes):
        scores = torch.einsum("pd,hd->ph", features[i], prototype)
        q = find_optimal_assignment(scores, 0.05, 20)
        # q = scores
        cluster_map = q.argmax(dim=-1)
        # cluster_map = cluster_map.view(eval_spatial_res, eval_spatial_res)
        cluster_maps.append(cluster_map)
    cluster_maps = torch.stack(cluster_maps)
    ## create one-hot cluster map
    reshaped_cluster_map = cluster_maps.reshape(-1, eval_spatial_res, eval_spatial_res)
    resized_cluster_maps = F.interpolate(reshaped_cluster_map.unsqueeze(1).float(), size=(input_size, input_size), mode="nearest")
    _, overlayed_data = overlay_video_cmap(resized_cluster_maps.squeeze(1), denormalized_data)
    wandb_image = wandb.Image(overlayed_data)
    logger.log({"cluster_map": wandb_image})


    cluster_maps = F.one_hot(cluster_maps, num_classes=head_num)
    cluster_maps = cluster_maps.permute(0, 2, 1)
    refined_prototypes = torch.einsum("bhp,bpd->bhd", cluster_maps.float(), attention_v)
    keys, _ = model.get_intermediate_layer_feats(data, feat="k", layer_num=-1)
    refined_cluster_maps = []
    # refined_prototypes = refined_prototypes / refined_prototypes.norm(dim=-1, keepdim=True)
    # keys = keys / keys.norm(dim=-1, keepdim=True)
    for i, prototype in enumerate(refined_prototypes):
        scores = torch.einsum("pd,hd->ph", keys[i], prototype)
        q = scores
        # q = scores
        cluster_map = q.argmax(dim=-1)
        cluster_map = cluster_map.view(eval_spatial_res, eval_spatial_res)
        refined_cluster_maps.append(cluster_map)
    refined_cluster_maps = torch.stack(refined_cluster_maps)
    refined_cluster_maps = F.interpolate(refined_cluster_maps.unsqueeze(1).float(), size=(input_size, input_size), mode="nearest")
    _, overlayed_data = overlay_video_cmap(refined_cluster_maps.squeeze(1), denormalized_data)
    wandb_image = wandb.Image(overlayed_data)
    logger.log({"refined_cluster_map": wandb_image})



def timet_dora(device, model, logger, data_loader):
    data, target = next(iter(data_loader))
    input_size = data.size(-1)
    denormalized_data = data * torch.tensor([0.229, 0.224, 0.255]).view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    denormalized_data = (denormalized_data * 255).type(torch.uint8)
    data = data.to(device)
    eval_spatial_res = model.spatial_resolution
    attn = model.get_last_selfattention(data)
    cls_attn = attn[:, :, 0, 1:]
    value_attention = attn[:, :, 1:, 1:]
    head_num = value_attention.size(1)
    ## normalize the attention heads between 0 and 1
    # cls_attn = cls_attn / cls_attn.sum(dim=-1, keepdim=True)

    queries, _ = model.get_intermediate_layer_feats(data, feat="q", layer_num=-1)
    keys, _ = model.get_intermediate_layer_feats(data, feat="k", layer_num=-1)
    values, _ = model.get_intermediate_layer_feats(data, feat="v", layer_num=-1)
    # queries = queries / queries.norm(dim=-1, keepdim=True)
    last_features = model.forward_features(data)
    prototypes = torch.einsum("bhp,bpd->bhd", cls_attn, last_features)
    ## normalize the prototypes
    prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)
    cluster_maps = []
    # features = model.forward_features(data)
    # features = features[:, 1:, :]
    features = last_features
    features = features / features.norm(dim=-1, keepdim=True)
    for i, prototype in enumerate(prototypes):
        scores = torch.einsum("pd,hd->ph", features[i], prototype)
        q = find_optimal_assignment(scores, 0.05, 20)
        # q = scores
        cluster_map = q.argmax(dim=-1)
        # cluster_map = cluster_map.view(eval_spatial_res, eval_spatial_res)
        cluster_maps.append(cluster_map)
    cluster_maps = torch.stack(cluster_maps)

    unmasked_features, ids_to_keep = model.masking_forward_features(data)
    unmasked_cluster_ids = cluster_maps[ids_to_keep]


    ## create one-hot cluster map




class TimeTDoRA(torch.nn.Module):
    def __init__(self, vit_model, num_prototypes=200, num_slots=0, itr=1, use_gru=False, use_kv=False, mask_ratio=None, logger=None, loss=None, use_registers=False, slot_source="backbone_features"):
        super(TimeTDoRA, self).__init__()
        self.feature_extractor = vit_model
        self.logger = logger
        self.mask_ratio = mask_ratio
        self.num_prototypes = num_prototypes
        self.criterion = torch.nn.CrossEntropyLoss()
        self.loss = loss
        self.mlp_head = torch.nn.Sequential(
            torch.nn.Linear(self.feature_extractor.embed_dim, 1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, 256),
        )
        # self.local_head = torch.nn.Sequential(
        #     torch.nn.Linear(self.feature_extractor.embed_dim, 512),
        #     torch.nn.GELU(),
        #     torch.nn.Linear(512, self.feature_extractor.embed_dim),
        # )
        # self.lc = torch.nn.Linear(self.feature_extractor.d_model, self.eval_spatial_resolution ** 2)
        self.freeze_feature_extractor(["blocks.11", "blocks.10"])
        prototype_init = torch.randn((num_prototypes, 256))
        prototype_init =  F.normalize(prototype_init, dim=-1, p=2)  
        self.prototypes = torch.nn.Parameter(prototype_init)
        self.use_registers = use_registers
        self.slot_source = slot_source
        if num_slots != 0:
            self.num_slots = num_slots
            if self.slot_source == "backbone_features":
                slot_dim = self.feature_extractor.embed_dim
            elif self.slot_source == "projected_features":
                slot_dim = 256
            self.slots = SlotAttention(num_slots=num_slots, dim=slot_dim, iters=itr, use_gru=use_gru, use_kv=use_kv)
        self.slot_source = slot_source
    

    def normalize_prototypes(self):
        with torch.no_grad():
            w = self.prototypes.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.prototypes.copy_(w)
    

    def freeze_feature_extractor(self, unfreeze_layers=[]):
        for name, param in self.feature_extractor.named_parameters():
            param.requires_grad = False
            for unfreeze_layer in unfreeze_layers:
                if unfreeze_layer in name:
                    param.requires_grad = True
                    break
            

    def train_step(self, batch_data):
        ## normalize the prototypes
        self.normalize_prototypes()
        if self.loss == "mask_consistency_global_prototype":
            clustering_loss = self.pipeline_2(batch_data)
            total_loss = mask_loss + clustering_loss
        elif self.loss == "mask_consistency":
            mask_loss = self.mask_consistency_loss(batch_data)
            total_loss = mask_loss
        elif self.loss == "global_prototype":
            clustering_loss = self.pipeline_2(batch_data)
            total_loss = clustering_loss
        else:
            raise NotImplementedError
        return total_loss

    def pipeline_1(self, batch_data):
        self.normalize_prototypes()
        bs, c, h, w = batch_data.shape
        # denormalized_video = denormalize_video(datum)
        features = self.feature_extractor.forward_features(batch_data) # (B, np, dim)
        if self.num_slots != 0:
            registers, global_prototypes = self.slots(features)
        else:
            attn = self.feature_extractor.get_last_selfattention(batch_data)
            cls_attn = attn[:, :, 0, 1:]
            global_prototypes = torch.einsum("bhp,bpd->bhd", cls_attn, features)
        global_prototypes = global_prototypes / global_prototypes.norm(dim=-1, keepdim=True)
        cluster_maps = []
        normalised_features = features / features.norm(dim=-1, keepdim=True)
        for i, prototype in enumerate(registers):
            scores = torch.einsum("pd,hd->ph", normalised_features[i], prototype)
            # q = find_optimal_assignment(scores, 0.05, 20)
            q = scores
            cluster_map = q.argmax(dim=-1)
            # cluster_map = cluster_map.view(eval_spatial_res, eval_spatial_res)
            cluster_maps.append(cluster_map)
        cluster_maps = torch.stack(cluster_maps)

        unmasked_features, ids_to_keep = self.feature_extractor.masking_forward_features(batch_data, mask_ratio=self.mask_ratio)
        unmasked_cluster_ids = cluster_maps[torch.arange(bs).unsqueeze(1), ids_to_keep].reshape(bs, ids_to_keep.size(-1))

        batch_scores, batch_q = self.extract_assignments(global_prototypes)
        unmasked_projected_features = self.mlp_head(unmasked_features)
        batch_unmasked_scores, batch_unmasked_q = self.extract_assignments(unmasked_projected_features)
        batch_unmasked_scores = batch_unmasked_scores.permute(0, 2, 1)
        batch_q = batch_q.argmax(dim=-1)
        gt_unmasked_q = batch_q[torch.arange(batch_q.size(0)).unsqueeze(1), unmasked_cluster_ids].reshape(bs, unmasked_cluster_ids.size(-1))
        clustering_loss = self.criterion(batch_unmasked_scores  / 0.1, gt_unmasked_q.long())
        contrastive_loss = self.local_contrastive_loss(registers, unmasked_features, unmasked_cluster_ids)
        return clustering_loss,contrastive_loss


    def pipeline_2(self, batch_data):
        bs, c, h, w = batch_data.shape
        # denormalized_video = denormalize_video(datum)
        features = self.feature_extractor.forward_features(batch_data) ## bs, np, dim
        prj_features = self.mlp_head(features) ## bs, np, dim
        if self.num_slots != 0:
            if self.slot_source == "backbone_features":
                registers, object_prototypes = self.slots(features)
            elif self.slot_source == "projected_features":
                registers, object_prototypes = self.slots(prj_features) ## bs, num_slots, dim
        else:
            raise NotImplementedError
        normalised_register = registers / registers.norm(dim=-1, keepdim=True)
        normalised_features = prj_features / prj_features.norm(dim=-1, keepdim=True)
        object_prototypes = object_prototypes / object_prototypes.norm(dim=-1, keepdim=True)
        if self.use_registers:
            if self.slot_source == "backbone_features":
                normalised_features = features / features.norm(dim=-1, keepdim=True)
            feature_objprto_sim = torch.einsum("bpd,bhd->bph", normalised_features, normalised_register)
        else:
            feature_objprto_sim = torch.einsum("bpd,bhd->bph", normalised_features, object_prototypes)
        objprto_prototype_sim = torch.einsum("bd,pd->bp", object_prototypes.flatten(0, 1), self.prototypes).reshape(bs, self.num_slots, self.num_prototypes)
        feature_prototype_indir_sim = torch.einsum("bpo,bol->bpl", feature_objprto_sim, objprto_prototype_sim)
        bs, np, _ = feature_prototype_indir_sim.shape
        batch_feature_q = find_optimal_assignment(feature_prototype_indir_sim.flatten(0, 1), 0.05, 10)
        batch_feature_q = batch_feature_q.reshape(bs, np, self.num_prototypes)
        batch_feature_q = batch_feature_q.argmax(dim=-1)

        feature_prototype_indir_sim = feature_prototype_indir_sim / 0.1

        batch_scores, batch_q = self.extract_assignments(prj_features)
        batch_q = batch_q.argmax(dim=-1)
        batch_scores = batch_scores / 0.1

        loss = self.criterion(feature_prototype_indir_sim.permute(0, 2, 1), batch_q.long())
        loss1 = self.criterion(batch_scores.permute(0, 2, 1), batch_feature_q.long())
        return loss + loss1


    def mask_consistency_loss(self, batch_data):
        bs, c, h, w = batch_data.shape
        # denormalized_video = denormalize_video(datum)
        features = self.feature_extractor.forward_features(batch_data) ## bs, np, dim
        prj_features = self.mlp_head(features) ## bs, np, dim
        batch_scores, batch_q = self.extract_assignments(prj_features)
        unmasked_features, ids_to_keep = self.feature_extractor.masking_forward_features(batch_data, mask_ratio=self.mask_ratio)
        unmasked_projected_features = self.mlp_head(unmasked_features)
        batch_unmasked_scores, batch_unmasked_q = self.extract_assignments(unmasked_projected_features)
        batch_unmasked_scores = batch_unmasked_scores.permute(0, 2, 1)
        batch_q = batch_q.argmax(dim=-1)
        batch_unmasked_q = batch_unmasked_q.argmax(dim=-1)
        gt_unmasked_q = batch_q[torch.arange(batch_q.size(0)).unsqueeze(1), ids_to_keep].reshape(bs, ids_to_keep.size(-1))
        unmasked_scores = batch_scores[torch.arange(batch_scores.size(0)).unsqueeze(1), ids_to_keep].reshape(bs, ids_to_keep.size(-1), self.num_prototypes)
        clustering_loss_1 = self.criterion(batch_unmasked_scores  / 0.1, gt_unmasked_q.long())
        clustering_loss_2 = self.criterion(unmasked_scores.permute(0, 2, 1), batch_unmasked_q.long())
        return clustering_loss_1 + clustering_loss_2

        

    def local_contrastive_loss(self, registers, unmasked_features, unmasked_cluster_ids):
        unmasked_features = self.local_head(unmasked_features)
        unmasked_features = unmasked_features / unmasked_features.norm(dim=-1, keepdim=True)
        similarity_matrix = torch.einsum("bpd,bhd->bph", unmasked_features, registers)
        similarity_matrix = similarity_matrix / 0.1
        similarity_matrix = similarity_matrix.permute(0, 2, 1)
        gt_unmasked = unmasked_cluster_ids
        loss = self.criterion(similarity_matrix, gt_unmasked.long())
        return loss


    def extract_assignments(self, projected_features):
        bs, np, dim = projected_features.shape
        projected_dim = projected_features.shape[-1]
        projected_features = projected_features.reshape(-1, projected_dim)
        normalized_projected_features = F.normalize(projected_features, dim=-1, p=2)

        batch_scores = torch.einsum('bd,nd->bn', normalized_projected_features , self.prototypes)
        batch_q = find_optimal_assignment(batch_scores, 0.05, 10)
        batch_q = batch_q.reshape(bs, np, self.num_prototypes)
        batch_scores = batch_scores.reshape(bs, np, self.num_prototypes)
        return batch_scores,batch_q
    

    def get_params_dict(self, model, exclude_decay=True, lr=1e-4):
        params = []
        excluded_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if exclude_decay and (name.endswith(".bias") or (len(param.shape) == 1)):
                    excluded_params.append(param)
                else:
                    params.append(param)
                print(f"{name} is trainable")
        return [{'params': params, 'lr': lr},
                    {'params': excluded_params, 'weight_decay': 0., 'lr': lr}]

    def get_optimization_params(self):
        feature_extractor_params = self.get_params_dict(self.feature_extractor,exclude_decay=True, lr=1e-5)
        mlp_head_params = self.get_params_dict(self.mlp_head,exclude_decay=True, lr=1e-4)
        prototypes_params = [{'params': self.prototypes, 'lr': 1e-4}]
        if self.num_slots != 0:
            slots_params = self.get_params_dict(self.slots, exclude_decay=True, lr=1e-4)
            all_params = feature_extractor_params + mlp_head_params + prototypes_params + slots_params
        else: 
            all_params = feature_extractor_params + mlp_head_params + prototypes_params
        return all_params


    def create_attn_visualization_wandb_dict(self, data):
        input_size = data.size(-1)
        denormalized_data = data.cpu() * torch.tensor([0.229, 0.224, 0.255]).view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        denormalized_data = (denormalized_data * 255).type(torch.uint8)
        eval_spatial_res = self.feature_extractor.spatial_resolution
        attn = self.feature_extractor.get_last_selfattention(data)
        cls_attn_prev = attn[:, :, 0, 1:]
        cls_attn = cls_attn_prev.reshape(-1, cls_attn_prev.size(1), eval_spatial_res, eval_spatial_res)
        cls_attn = F.interpolate(cls_attn, size=(input_size, input_size), mode="nearest")
        for j, d in enumerate(denormalized_data):
            # Create a subplot for each attention head and the original image
            fig, axs = plt.subplots(4, 4, figsize=(12, 12))  # You can adjust the figsize as needed

            # Plot the original image
            axs[0, 0].imshow(d.permute(1, 2, 0))
            axs[0, 0].axis('off')  # Optionally turn off axis

            # Plot each attention map
            for i in range(1, 13):
                axs[i // 4, i % 4].imshow(cls_attn[j, i - 1].cpu().detach().numpy())
                axs[i // 4, i % 4].axis('off')  # Optionally turn off axis

            # Remove empty subplots
            for ax in axs.flat:
                if not ax.images:
                    fig.delaxes(ax)

            fig.tight_layout(pad=1.0)  # Adjust layout

            ## create the histogram of the attention heads
            fig1, axs1 = plt.subplots(4, 3, figsize=(12, 12))
            for i in range(0, 12):
                axs1[i // 3, i % 3].hist(cls_attn_prev[j, i - 1].cpu().detach().numpy().reshape(-1), bins=10)
                axs1[i // 3, i % 3].set_title(f"Attention Map {i}")
                axs1[i // 3, i % 3].set_xlim(0, 1)
            
            
            fig1.tight_layout(pad=1.0)


            # Create wandb Image
            wandb_image = wandb.Image(fig, caption=f"Attention Map {j}")
            wandb_image1 = wandb.Image(fig1, caption=f"Attention Map {j}_hist")

            plt.close(fig) 
            plt.close(fig1)

            return {"attention_map": wandb_image, "attention_map_hist": wandb_image1}
    
    def create_feature_slot_visualization_wandb_dict(self, data):
        bs, c, h, w = data.shape
        denormalized_data = data.cpu() * torch.tensor([0.229, 0.224, 0.255]).view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        denormalized_data = (denormalized_data * 255).type(torch.uint8)
        features = self.feature_extractor.forward_features(data) ## bs, np, dim
        prj_features = self.mlp_head(features) ## bs, np, dim
        if self.num_slots != 0:
            if self.slot_source == "backbone_features":
                registers, object_prototypes = self.slots(features)
            elif self.slot_source == "projected_features":
                registers, object_prototypes = self.slots(prj_features) ## bs, num_slots, dim
        else:
            raise NotImplementedError

        normalised_register = registers / registers.norm(dim=-1, keepdim=True)
        prj_features = prj_features / prj_features.norm(dim=-1, keepdim=True)
        object_prototypes = object_prototypes / object_prototypes.norm(dim=-1, keepdim=True)
        if self.use_registers:
            if self.slot_source == "backbone_features":
                normalised_features = features / features.norm(dim=-1, keepdim=True)
            elif self.slot_source == "projected_features":
                normalised_features = prj_features
            feature_objprto_sim = torch.einsum("bpd,bhd->bph", normalised_features, normalised_register)
        else:
            feature_objprto_sim = torch.einsum("bpd,bhd->bph", prj_features, object_prototypes)
        cluster_map = feature_objprto_sim.argmax(dim=-1)
        cluster_map = cluster_map.view(bs, self.feature_extractor.spatial_resolution, self.feature_extractor.spatial_resolution)
        cluster_map = F.interpolate(cluster_map.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1)
        _, overlayed_data = overlay_video_cmap(cluster_map.squeeze(1), denormalized_data)
        wandb_image = wandb.Image(overlayed_data)
        return {"cluster_map": wandb_image}


    def validate_step(self, img):
        self.feature_extractor.eval()
        with torch.no_grad():
            spatial_features = self.feature_extractor.forward_features(img)  # (B, np, dim)
        return spatial_features

    def save(self, path):
        torch.save(self.state_dict(), path)


class TimeTDoRATrainer():
    def __init__(self, train_dataloader, test_dataloader, time_tuning_model, num_epochs, device, logger):
        self.dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.time_tuning_model = time_tuning_model
        self.device = device
        self.time_tuning_model = self.time_tuning_model.to(self.device)
        self.optimizer = None
        self.num_epochs = num_epochs
        self.logger = logger
        self.logger.watch(time_tuning_model, log="all", log_freq=10)
    
    
    def setup_optimizer(self, optimization_config):
        model_params = self.time_tuning_model.get_optimization_params()
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
        self.optimizer = TimeTv2Optimizer(model_params, init_lr, peak_lr, warmup_steps, grad_norm_clip, max_itr)
        self.optimizer.setup_optimizer()
        self.optimizer.setup_scheduler()
    

    def train_one_epoch(self):
        self.time_tuning_model.train()
        epoch_loss = 0
        before_loading_time = time.time()
        for i, batch in enumerate(self.dataloader):
            after_loading_time = time.time()
            print("Loading Time: {}".format(after_loading_time - before_loading_time))
            datum, annotations = batch
            # annotations = annotations.squeeze(1)
            # datum = datum.squeeze(1)
            datum = datum.to(self.device)
            clustering_loss = self.time_tuning_model.train_step(datum)
            total_loss = clustering_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.optimizer.update_lr()
            epoch_loss += total_loss.item()
            print("Iteration: {} Loss: {}".format(i, total_loss.item()))
            self.logger.log({"clustering_loss": clustering_loss.item()})
            lr = self.optimizer.get_lr()
            self.logger.log({"lr": lr})
            before_loading_time = time.time()
        epoch_loss /= (i + 1)
        # if epoch_loss < 2.5:
        #     self.time_tuning_model.save(f"Temp/model_{epoch_loss}.pth")
        print("Epoch Loss: {}".format(epoch_loss))
    
    def train(self):
        for epoch in range(self.num_epochs):
            print("Epoch: {}".format(epoch))
            if epoch % 1 == 0:
                self.validate(epoch)
            self.train_one_epoch()
            # self.validate(epoch)
            # self.patch_prediction_model.save_model(epoch)
            # self.validate(epoch)
    
    def validate(self, epoch, val_spatial_resolution=56):
        self.time_tuning_model.eval()
        with torch.no_grad():
            metric = PredsmIoU(21, 21)
            # spatial_feature_dim = self.model.get_dino_feature_spatial_dim()
            spatial_feature_dim = 50
            clustering_method = PerDatasetClustering(spatial_feature_dim, 21)
            feature_spatial_resolution = self.time_tuning_model.feature_extractor.spatial_resolution
            feature_group = []
            targets = []
            for i, (x, y) in enumerate(self.test_dataloader):
                target = (y * 255).long()
                img = x.to(self.device)
                if i <= 2:
                    self.visualize_feature_slot_assignment(img)
                spatial_features = self.time_tuning_model.validate_step(img)  # (B, np, dim)
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
        self.time_tuning_model.eval()
        losses = []
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_dataloader):
                target = (y * 255).long()
                img = x.to(self.device)
                loss = self.time_tuning_model.validate_step1(img)  # (B, np, dim)
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
                    self.time_tuning_model.save(f"Temp/model_{epoch}_{best_miou}.pth")
            self.train_one_epoch()
            # self.validate(epoch)
            # self.patch_prediction_model.save_model(epoch)
            # self.validate(epoch)

    def lc_validation(self, train_dataloader, val_dataloader, device):
        self.time_tuning_model.eval()
        model = self.time_tuning_model.feature_extractor
        lc_module = LinearFinetuneModule(model, train_dataloader, val_dataloader, device)
        final_miou = lc_module.linear_segmentation_validation()
        self.logger.log({"lc_val_miou": final_miou})
        return final_miou

    def visualize_attention_heads(self, data):
        self.time_tuning_model.eval()
        self.time_tuning_model.to(self.device)
        with torch.no_grad():
            visualization_dict = self.time_tuning_model.create_attn_visualization_wandb_dict(data)
            self.logger.log(visualization_dict)

    def visualize_feature_slot_assignment(self, data):
        self.time_tuning_model.eval()
        self.time_tuning_model.to(self.device)
        with torch.no_grad():
            visualization_dict = self.time_tuning_model.create_feature_slot_visualization_wandb_dict(data)
            self.logger.log(visualization_dict)



    
def run(args):
    patch_size = args.patch_size
    backbone = args.backbone
    use_registers = args.use_registers
    num_head = args.num_head
    input_size = args.input_size
    device = args.device
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    spatial_resolution = input_size // patch_size
    num_workers = args.num_workers
    mask_ratio = args.mask_ratio
    num_slots = args.num_slots
    itr = args.itr
    use_gru = args.use_gru
    augmentation = args.augmentation
    use_kv = args.use_kv
    wandb_mode = args.wandb
    dataset_name = args.dataset
    num_prototypes = args.num_prototypes
    slot_source = args.slot_source
    loss = args.loss
    torch.autograd.set_detect_anomaly(True)
    experiment_name = f"dataset:{dataset_name}_patch_size:{patch_size}_backbone:{backbone}_slot_source:{slot_source}_use_registers:{use_registers}_num_prototypes:{num_prototypes}_itr:{itr}_use_gru:{use_gru}_use_kv:{use_kv}_num_head:{num_head}_input_size:{input_size}_mask_ratio:{mask_ratio}_num_slots:{num_slots}_augmentation:{augmentation}_loss:{loss}"
    config = vars(args)
    if backbone == "dino_v1":
        model = MyVisionTransformer(patch_size=patch_size, img_size=[input_size], embed_dim=384, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        # model = vit_base(patch_size=patch_size)
        pretraining = torch.hub.load('facebookresearch/dino:main', f'dino_vits{patch_size}')
    elif backbone == "vit":
        model = MyVisionTransformer(img_size=[input_size], embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        pretraining = timm.create_model(f'vit_base_patch{patch_size}_224', pretrained=True)
    elif backbone == "dino_v2":
        if use_registers:
            model = MyVisionTransformer(reg_tokens=4, spatial_resolution=spatial_resolution, patch_size=patch_size, img_size=input_size)
            pretraining = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=True)
        else:
            model = MyVisionTransformer(spatial_resolution=spatial_resolution, patch_size=patch_size, img_size=input_size)
            pretraining = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True)

    model.spatial_resolution = spatial_resolution
    msg = model.load_state_dict(pretraining.state_dict(), strict=False)
    print(msg)
    model.to(device)
    today = date.today()
    d1 = today.strftime("%d_%m_%Y")
    project_name = "DoRA_TimeT"
    logger = wandb.init(project=project_name, mode=wandb_mode, group=d1, job_type="TimeT_DoRA_Pascal", name=experiment_name, config=config, dir="/ssdstore/ssalehi/timet_logst")
    ## figure to show the image and each 12 head of the attention map


    min_scale_factor = 0.5
    max_scale_factor = 1.0

    # Create the transformation

    color_jitter = trn.ColorJitter(
        0.8, 0.8, 0.8,
        0.2 
    )
    color_transform = [trn.RandomApply([color_jitter], p=0.8),
                        trn.RandomGrayscale(p=0.2)]
    blur = GaussianBlur(sigma=[0.1, 2.0])
    color_transform.append(trn.RandomApply([blur], p=0.5))
    color_transform = trn.Compose(color_transform)

    # Construct final transforms
    normalize = trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_train_transform = trn.Compose([color_transform, trn.RandomResizedCrop((224, 224), (0.25, 1)), trn.ToTensor(), normalize])
    # image_train_transform = trn.Compose([
    #     trn.ToTensor(),
    #     trn.RandomHorizontalFlip(),
    #     trn.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
    #     trn.
    #     trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])
    # ])

    shared_transform = Compose([
        Resize(size=(input_size, input_size)),
    ])

    train_transforms = {"img": image_train_transform, "target": None, "shared": shared_transform}
    if dataset_name == "pascal":
        train_dataset = PascalVOCDataModule(batch_size=batch_size, train_transform=train_transforms, val_transform=train_transforms, test_transform=train_transforms)
    elif dataset_name == "coco":
        train_dataset = CocoDataModule(batch_size=batch_size, train_transform=train_transforms, val_transform=train_transforms, test_transform=train_transforms, img_dir="/ssdstore/ssalehi/coco/train2017", annotation_dir="/ssdstore/ssalehi/coco/annotations/instances_train2017.json", num_workers=num_workers)
    elif dataset_name == "imagenet_100":
        train_dataset = ImangeNet_100_Handler(batch_size=batch_size, dataset_path="/ssdstore/ssalehi/imagenet-100/imgs", transformations=train_transforms, val_transformations=train_transforms, num_workers=num_workers)
    train_dataset.setup()
    train_dataloader = train_dataset.get_train_dataloader()
    timet_dora = TimeTDoRA(model, logger=logger, num_prototypes=num_prototypes, mask_ratio=mask_ratio, itr=itr, use_gru=use_gru, use_kv=use_kv, num_slots=num_slots, loss=loss, use_registers=use_registers, slot_source=slot_source)
    timet_dora = timet_dora.to(device)
    optimization_config = {
        'init_lr': 1e-4,
        'peak_lr': 1e-3,
        'decay_half_life': 0,
        'warmup_steps': 0,
        'grad_norm_clip': 0,
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
    TimeTDoRA_trainer = TimeTDoRATrainer(train_dataloader, test_dataloader, timet_dora, num_epochs, device, logger)
    TimeTDoRA_trainer.setup_optimizer(optimization_config)
    TimeTDoRA_trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--backbone', type=str, default="dino_v1")
    parser.add_argument('--use_registers', type=bool, default=True)
    parser.add_argument('--num_head', type=int, default=-1)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--num_prototypes', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mask_ratio', type=float, default=0.)
    parser.add_argument('--num_slots', type=int, default=6)
    parser.add_argument('--dataset', type=str, default="imagenet_100", choices=["pascal", "coco", "imagenet_100"])
    parser.add_argument('--slot_source', type=str, default="backbone_features")
    parser.add_argument('--augmentation', type=str, default="everything+crop")
    parser.add_argument('--loss', type=str, default="global_prototype", choices=["mask_consistency_global_prototype", "mask_consistency", "global_prototype"])
    parser.add_argument('--use_gru', type=bool, default=True)
    parser.add_argument('--itr', type=int, default=3)
    parser.add_argument('--use_kv', type=bool, default=False)
    parser.add_argument('--device', type=str, default="cuda:6")
    parser.add_argument('--wandb', type=str, default="online")
    args = parser.parse_args()
    run(args)
   
