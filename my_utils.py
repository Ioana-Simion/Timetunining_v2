import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from torchvision.transforms import GaussianBlur
from typing import List
from IPython.display import display, display_markdown
import io
import os, sys
import requests
import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import glob
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
from typing import List
from torchvision.utils import draw_segmentation_masks
import cv2


def show_trainable_paramters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)



def process_attentions(attn_batch, spatial_res, threshold = 0.5, blur_sigma = 0.6):
    """
    Process [0,1] attentions to binary 0-1 mask. Applies a Guassian filter, keeps threshold % of mass and removes
    components smaller than 3 pixels.
    The code is adapted from https://github.com/facebookresearch/dino/blob/main/visualize_attention.py but removes the
    need for using ground-truth data to find the best performing head. Instead we simply average all head's attentions
    so that we can use the foreground mask during training time.
    :param attentions: torch 4D-Tensor containing the averaged attentions
    :param spatial_res: spatial resolution of the attention map
    :param threshold: the percentage of mass to keep as foreground.
    :param blur_sigma: standard deviation to be used for creating kernel to perform blurring.
    :return: the foreground mask obtained from the ViT's attention.
    """
    # Blur attentions
    # attns_processed = torch.cat(attns_group, dim = 0)
    attns_processed = sum(attn_batch[:, i] * 1 / attn_batch.size(1) for i in range(attn_batch.size(1)))
    attentions = attns_processed.reshape(-1, 1, spatial_res, spatial_res)
    attentions = GaussianBlur(7, sigma=(blur_sigma))(attentions)
    attentions = attentions.reshape(attentions.size(0), 1, spatial_res ** 2)
    # Keep threshold% of mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=-1, keepdim=True)
    cumval = torch.cumsum(val, dim=-1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    th_attn[:, 0] = torch.gather(th_attn[:, 0], dim=1, index=idx2[:, 0])
    th_attn = th_attn.reshape(attentions.size(0), 1, spatial_res, spatial_res).float()
    # Remove components with less than 3 pixels
    for j, th_att in enumerate(th_attn):
        labelled = label(th_att.cpu().numpy())
        for k in range(1, np.max(labelled) + 1):
            mask = labelled == k
            if np.sum(mask) <= 2:
                th_attn[j, 0][mask] = 0
    return th_attn.detach()



def preprocess(imgs):
    img_group = []
    for i in range(imgs.shape[0]):
        img = imgs[i]
        img = T.ToPILImage()(img.cpu())
        target_image_size = 224
        s = min(img.size)
        
        if s < target_image_size:
            raise ValueError(f'min dim for image {s} < {target_image_size}')
            
        r = target_image_size / s
        s = (round(r * img.size[1]), round(r * img.size[0]))
        img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
        img = TF.center_crop(img, output_size=2 * [target_image_size])
        img = torch.unsqueeze(T.ToTensor()(img), 0)
        img_group.append(map_pixels(img))
    return torch.cat(img_group, dim = 0)



def cosine_scheduler(base_value: float, final_value: float, epochs: int, niter_per_ep: int):
    # Construct cosine schedule starting at base_value and ending at final_value with epochs * niter_per_ep values.
    iters = np.arange(epochs * niter_per_ep)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def denormalize_video(video):
    """
    video: [1, nf, c, h, w]
    """
    denormalized_video = video.cpu().detach() * torch.tensor([0.225, 0.225, 0.225]).view(1, 1, 3, 1, 1) + torch.tensor([0.45, 0.45, 0.45]).view(1, 1, 3, 1, 1)
    denormalized_video = (denormalized_video * 255).type(torch.uint8)
    denormalized_video = denormalized_video.squeeze(0)
    return denormalized_video

def overlay_video_cmap(cluster_maps, denormalized_video):
    """
    cluster_maps: [1, nf, h, w]
    denormalized_video: [1, nf, c, h, w]
    """
    cluster_maps = cluster_maps.squeeze(0)
        ## convert cluster_maps to [num_maps, h, w]
    masks = []
    cluster_ids = torch.unique(cluster_maps)
    for cluster_map in cluster_maps:
        mask = torch.zeros((cluster_ids.shape[0], cluster_map.shape[0], cluster_map.shape[1])) 
        mask = mask.type(torch.bool)
        for i, cluster_id in enumerate(cluster_ids):
                ## make a boolean mask for each cluster
                ## make a boolean mask for each cluster if cluster_map == cluster_id
            boolean_mask = (cluster_map == cluster_id)
            mask[i, :, :] = boolean_mask
        masks.append(mask)
    cluster_maps = torch.stack(masks)
            
    overlayed = [
                draw_segmentation_masks(img, masks=mask, alpha=0.7)
                for img, mask in zip(denormalized_video, cluster_maps)
            ]
    overlayed = torch.stack(overlayed)
    return cluster_maps,overlayed


def overlay(image, mask, color = (255, 0, 0), alpha = 0.5, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined
