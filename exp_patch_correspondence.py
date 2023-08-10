import torch
from torchvision import transforms
from pytorchvideo.data import Ucf101, make_clip_sampler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models import FeatureExtractor
from my_utils import overlay, denormalize_video
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip, 
    Resize
)
import wandb
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

project_name = "TimeTuning_v2"
## generate ListeColorMap of distinct colors

## what are the colors for red, blue, green, brown, yello, orange, purple, white, black, maroon, olive, teal, navy, gray, silver
## Fill the ListedColormap with the colors above

cmap = ListedColormap(['#FF0000', '#0000FF', '#008000', '#A52A2A', '#FFFF00', '#FFA500', '#800080', '#FFFFFF', '#000000', '#800000', '#808000', '#008080', '#000080', '#808080', '#C0C0C0'])

def generate_random_crop(img, crop_size):
    c, h, w = img.shape
    crop = torch.zeros((h, w))
    x = torch.randint(0, h - crop_size, (1,)).item()
    y = torch.randint(0, w - crop_size, (1,)).item()
    crop[x:x + crop_size, y:y + crop_size] = 1
    return crop


class CorrespondenceDetection():
    def __init__(self, feature_extractor, window_szie) -> None:
        self.feature_extractor = feature_extractor
        self.spatial_resolution = feature_extractor.eval_spatial_resolution
        self.d_model = feature_extractor.d_model
        self.window_size = window_szie

    def __call__(self, img1, img2, crop):
        c, h, w = img1.shape
        patch_size = h // self.spatial_resolution
        crop = crop.reshape(h // patch_size, patch_size, w // patch_size, patch_size).permute(0, 2, 1, 3)
        crop = crop.flatten(2, 3)
        croped_feature_mask = crop.sum(-1) > 0 ## size (bs, spatial_resolution, spatial_resolution)
        ## find the idx of the croped features_mask
        img1_features, img1_attention = self.feature_extractor.forward_features(img1.unsqueeze(0))
        img1_features = img1_features.reshape(self.spatial_resolution, self.spatial_resolution, self.d_model)
        img2_features, img2_attention = self.feature_extractor.forward_features(img2.unsqueeze(0))
        img2_features = img2_features.reshape(self.spatial_resolution, self.spatial_resolution, self.d_model)
        similarity = torch.einsum('xyd,kzd->xykz', img1_features, img2_features)
        croped_feature_idx = croped_feature_mask.nonzero()
        most_similar_features_mask = torch.zeros(self.spatial_resolution, self.spatial_resolution)
        revised_crop = torch.zeros(self.spatial_resolution, self.spatial_resolution)
        for i, mask_idx in enumerate(croped_feature_idx):
            print(mask_idx)
            revised_crop[mask_idx[0], mask_idx[1]] = i + 1
            min_x, max_x = max(0, mask_idx[0] - self.window_size), min(self.spatial_resolution, mask_idx[0] + self.window_size)
            min_y, max_y = max(0, mask_idx[1] - self.window_size), min(self.spatial_resolution, mask_idx[1] + self.window_size)
            neiborhood_similarity = similarity[mask_idx[0], mask_idx[1], min_x:max_x, min_y:max_y]
            max_valuev = neiborhood_similarity.max()
            indices = (neiborhood_similarity == max_valuev).nonzero()[0]
            most_similar_features_mask[indices[0] + min_x, indices[1] + min_y] = i + 1
        resized_most_similar_features_mask = torch.nn.functional.interpolate(most_similar_features_mask.unsqueeze(0).unsqueeze(0), size=(h, w), mode='nearest').squeeze(0).squeeze(0)
        resized_revised_crop = torch.nn.functional.interpolate(revised_crop.unsqueeze(0).unsqueeze(0), size=(h, w), mode='nearest').squeeze(0).squeeze(0)
        return resized_most_similar_features_mask, resized_revised_crop


if __name__ == "__main__":
    device = "cuda:0"
    ucf101_path = '/ssdstore/ssalehi/ucf101/data/UCF101'
    clip_durations = 2
    logger = wandb.init(project=project_name, group='exp_patch_correspondence', job_type='debug')
    train_transform = Compose(
        [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                UniformTemporalSubsample(8),
                Lambda(lambda x: x / 255.0),
                Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                RandomShortSideScale(min_size=256, max_size=320),
                Resize((224, 224)),
                RandomHorizontalFlip(p=0.5),
                ]
            ),
            ),
        ]
    )
    train_dataset = Ucf101(
        data_path=ucf101_path,
        clip_sampler=make_clip_sampler("random", clip_durations),
        decode_audio=False,
        transform=train_transform,
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=1,  # Adjust the batch size according to your system's capacity
        num_workers=4,  # Adjust the number of workers based on your system's capacity
        pin_memory=True,
    )


input_size = 224
eval_spatial_resolution = input_size // 16
vit_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
feature_extractor = FeatureExtractor(vit_model, eval_spatial_resolution=eval_spatial_resolution)
feature_extractor = feature_extractor.to(device)
CorDet = CorrespondenceDetection(feature_extractor, window_szie=2)

for i, batch in enumerate(dataloader):
    video =  batch["video"]
    video = video.permute(0, 2, 1, 3, 4)
    video = video.to(device)
    video = video.squeeze(0)
    crop = generate_random_crop(video[0], 64)
    img1 = video[0]
    img2 = video[3]
    ## overlay crop on the image 2 and log it
    plt.imshow(img1.permute(1, 2, 0).detach().cpu().numpy())
    plt.imshow(crop.cpu().numpy(), alpha=0.5)
    plt.savefig("Temp/overlaied_img1.png")
    denormalized_video = denormalize_video(video)
    overlaied_img2 = overlay(img2.permute(1, 2, 0).detach().cpu().numpy(), crop.cpu().numpy())
    overlaied_img2 = torch.from_numpy(overlaied_img2).permute(2, 0, 1)
    # wandb.log({"overlaied_img2": wandb.Image(overlaied_img2)})
    sailiancy, revised_crop = CorDet(img1, img2, crop)
    plt.imshow(img1.permute(1, 2, 0).detach().cpu().numpy())
    plt.imshow(revised_crop.cpu().numpy(), alpha=0.5, cmap=cmap)
    plt.savefig("Temp/revised_crop_img1.png")
    img1 = denormalized_video[0]
    img2 = denormalized_video[3]
    plt.imshow(img2.permute(1, 2, 0).detach().cpu().numpy())
    plt.imshow(sailiancy.detach().cpu().numpy(), alpha=0.5, cmap=cmap)
    plt.savefig("Temp/overlaied_img2.png")
    overlaied_img1 = overlay(img1.permute(1, 2, 0).detach().cpu().numpy(), sailiancy.detach().cpu().numpy())
    overlaied_img1 = torch.from_numpy(overlaied_img1).permute(2, 0, 1)
    # wandb.log({"overlaied_img1": wandb.Image(overlaied_img1)})

        
