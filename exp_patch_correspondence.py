import torch
from torchvision import transforms
from pytorchvideo.data import Ucf101, make_clip_sampler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from clustering import PerDatasetClustering
from data_loader import PascalVOCDataModule
from eval_metrics import PredsmIoU
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
    Lambda,
    RandomCrop,
    RandomHorizontalFlip, 
)
import wandb
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from optimizer import PatchCorrespondenceOptimizer
import torchvision.transforms as trn

from transformations import Compose, Resize

project_name = "TimeTuning_v2"
## generate ListeColorMap of distinct colors

## what are the colors for red, blue, green, brown, yello, orange, purple, white, black, maroon, olive, teal, navy, gray, silver
## Fill the ListedColormap with the colors above

cmap = ListedColormap(['#FF0000', '#0000FF', '#008000', '#A52A2A', '#FFFF00', '#FFA500', '#800080', '#FFFFFF', '#000000', '#800000', '#808000', '#008080', '#000080', '#808080', '#C0C0C0'])

def generate_random_crop(img, crop_size):
    bs, c, h, w = img.shape
    crop = torch.zeros((bs, h, w))
    x = torch.randint(0, h - crop_size, (1,)).item()
    y = torch.randint(0, w - crop_size, (1,)).item()
    crop[:, x:x + crop_size, y:y + crop_size] = 1
    return crop


class CorrespondenceDetection():
    def __init__(self, window_szie, spatial_resolution=14) -> None:
        self.window_size = window_szie
        self.neihbourhood = self.restrict_neighborhood(spatial_resolution, spatial_resolution, self.window_size)

    
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
        bs, spatial_resolution, spatial_resolution, d_model = features1.shape
        _, h, w = crops.shape
        patch_size = h // spatial_resolution
        crops = crops.reshape(bs, h // patch_size, patch_size, w // patch_size, patch_size).permute(0, 1, 3, 2, 4)
        crops = crops.flatten(3, 4)
        croped_feature_mask = crops.sum(-1) > 0 ## size (bs, spatial_resolution, spatial_resolution)
        ## find the idx of the croped features_mask
        features1 = F.normalize(features1, dim=-1)
        features2 = F.normalize(features2, dim=-1)
        similarities = torch.einsum('bxyd,bkzd->bxykz', features1, features2)
        most_similar_features_mask = torch.zeros(bs, spatial_resolution, spatial_resolution)
        revised_crop = torch.zeros(bs, spatial_resolution, spatial_resolution)
        similarities = similarities * self.neihbourhood.unsqueeze_(0)
        for i, similarity in enumerate(similarities):
            croped_feature_idx = croped_feature_mask[i].nonzero()
            for j, mask_idx in enumerate(croped_feature_idx):
                # print(mask_idx)
                revised_crop[i, mask_idx[0], mask_idx[1]] = 1
                min_x, max_x = max(0, mask_idx[0] - self.window_size), min(spatial_resolution, mask_idx[0] + self.window_size)
                min_y, max_y = max(0, mask_idx[1] - self.window_size), min(spatial_resolution, mask_idx[1] + self.window_size)
                neiborhood_similarity = similarity[mask_idx[0], mask_idx[1], min_x:max_x, min_y:max_y]
                max_value = neiborhood_similarity.max()
                indices = (neiborhood_similarity == max_value).nonzero()[0]
                label_patch_number = (indices[0] + min_x) * spatial_resolution + (indices[1] + min_y)
                most_similar_features_mask[i, mask_idx[0], mask_idx[1]] = label_patch_number

        resized_most_similar_features_mask = torch.nn.functional.interpolate(most_similar_features_mask.unsqueeze(0), size=(h, w), mode='nearest').squeeze(0)
        resized_revised_crop = torch.nn.functional.interpolate(revised_crop.unsqueeze(0), size=(h, w), mode='nearest').squeeze(0)
        return resized_most_similar_features_mask, resized_revised_crop
    


class PatchPredictionModel(torch.nn.Module):
    def __init__(self, input_size, vit_model, prediction_window_size=2, logger=None):
        super(PatchPredictionModel, self).__init__()
        self.input_size = input_size
        self.eval_spatial_resolution = input_size // 16
        self.feature_extractor = FeatureExtractor(vit_model, eval_spatial_resolution=self.eval_spatial_resolution)
        self.feature_extractor = self.feature_extractor.to(device)
        self.prediction_window_size = prediction_window_size
        self.CorDet = CorrespondenceDetection(window_szie=self.prediction_window_size)
        self.device = device
        self.logger = logger
        self.criterion = torch.nn.CrossEntropyLoss()
        self.key_head = torch.nn.Linear(self.feature_extractor.d_model, self.feature_extractor.d_model)
        self.query_head = torch.nn.Linear(self.feature_extractor.d_model, self.feature_extractor.d_model)
        self.value_head = torch.nn.Linear(self.feature_extractor.d_model, self.feature_extractor.d_model)
        self.lc = torch.nn.Linear(self.feature_extractor.d_model, self.eval_spatial_resolution ** 2)
        self.feature_extractor.freeze_feature_extractor(["blocks.11", "blocks.10"])
        self.cross_attention_layer = torch.nn.MultiheadAttention(embed_dim=self.feature_extractor.d_model, num_heads=1, batch_first=True)
    
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
        mask = torch.zeros(bs, np).to(self.device)
        ids = torch.randperm(np)[:int(np * percentage)]
        mask[:, ids] = 1
        mask = mask.unsqueeze(-1).repeat(1, 1, d_model)
        features = features * mask
        return features
    
    def cross_attention(self, query, key, value, num_heads=1):
        """
        query: [bs, nq, d_model]
        key: [bs, np, d_model]
        value: [bs, np, d_model]

        return: [bs, nq, d_model]
        """
        # Parameters
        embedding_dim = query.shape[-1]
        output, attention_weights = self.cross_attention_layer(query, key, value)
        return output

    
    def train_step(self, imgs1, imgs2, crop_size):
        bs = imgs1.shape[0]
        crop = generate_random_crop(imgs1, crop_size)
        img1_features, img2_features = self.forward(imgs1, imgs2)
        sailiancy, revised_crop = self.CorDet(img1_features, img2_features, crop)
        ## find the szie of revised_crop where the value is not 0
        idxs = (revised_crop != 0).nonzero()
        min_x, max_x = idxs[:, 1].min(), idxs[:, 1].max()
        min_y, max_y = idxs[:, 2].min(), idxs[:, 2].max()
        h = max_x - min_x + 1
        w = max_y - min_y + 1
        crop_mask = revised_crop > 0
        ## select the cropped area and the corresponding labels 
        cropped_area = imgs1[crop_mask.unsqueeze(1).repeat(1, 3, 1, 1)]
        croped_labels = sailiancy[crop_mask]
        cropped_area = cropped_area.reshape(bs, 3, h, w)
        croped_labels = croped_labels.reshape(bs, h, w)
        ## resize cropped_area to (bs, 3, 224, 224) and croped_labels to (bs, 224, 224)
        cropped_area = torch.nn.functional.interpolate(cropped_area, size=(96, 96), mode='bilinear')
        croped_labels = torch.nn.functional.interpolate(croped_labels.unsqueeze(1), size=(96, 96), mode='nearest').squeeze(1)
        croped_labels = croped_labels.long().to(self.device)
        masked_features2 = self.mask_features(img2_features.flatten(1, 2))
        cropped_area_features, _ = self.feature_extractor.forward_features(cropped_area) ## size (bs, 36, d_model)
        cross_attented_features = self.cross_attention(self.query_head(cropped_area_features), self.key_head(masked_features2), self.value_head(masked_features2)) ## size (bs, 36, d_model)
        predictions = self.lc(cross_attented_features) ## size (bs, 36, 196)
        predictions = predictions.reshape(bs, 6, 6, 196).permute(0, 3, 1, 2) ## size (bs, 196, 6, 6)
        predictions = torch.nn.functional.interpolate(predictions, size=(96, 96), mode='bilinear')
        loss = self.criterion(predictions, croped_labels)
        return loss
    

    def get_optimization_params(self):
        return [
            {"params": self.feature_extractor.parameters(), "lr": 1e-5},
            {"params": self.key_head.parameters(), "lr": 1e-4},
            {"params": self.query_head.parameters(), "lr": 1e-4},
            {"params": self.value_head.parameters(), "lr": 1e-4},
            {"params": self.lc.parameters(), "lr": 1e-4},
        ]


    def visualize(self, img1, img2, crops):
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
        plt.imshow(dn_img1.permute(1, 2, 0).detach().cpu().numpy())
        plt.imshow(crop.cpu().numpy(), alpha=0.5)
        plt.savefig("Temp/overlaied_img1.png")
        overlaied_img2 = overlay(dn_img2.permute(1, 2, 0).detach().cpu().numpy(), crop.cpu().numpy())
        overlaied_img2 = torch.from_numpy(overlaied_img2).permute(2, 0, 1)
        # wandb.log({"overlaied_img2": wandb.Image(overlaied_img2)})
        plt.imshow(dn_img1.permute(1, 2, 0).detach().cpu().numpy())
        plt.imshow(revised_crop.cpu().numpy(), alpha=0.5, cmap=cmap)
        plt.savefig("Temp/revised_crop_img1.png")
        plt.imshow(dn_img2.permute(1, 2, 0).detach().cpu().numpy())
        plt.imshow(sailiancy.detach().cpu().numpy(), alpha=0.5, cmap=cmap)
        plt.savefig("Temp/overlaied_img2.png")
        overlaied_img1 = overlay(dn_img1.permute(1, 2, 0).detach().cpu().numpy(), sailiancy.detach().cpu().numpy())
        overlaied_img1 = torch.from_numpy(overlaied_img1).permute(2, 0, 1)
        # wandb.log({"overlaied_img1": wandb.Image(overlaied_img1)})


    def validate_step(self, img):
        self.feature_extractor.eval()
        with torch.no_grad():
            spatial_features, _ = self.feature_extractor.forward_features(img)  # (B, np, dim)
        return spatial_features

        





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
    
    def visualize(self):
        for i, batch in enumerate(self.dataloader):
            video = batch['video']
            video = video.permute(0, 2, 1, 3, 4)
            imgs1, imgs2 = video[:, 0], video[:, 1]
            imgs1, imgs2 = imgs1.to(self.device), imgs2.to(self.device)
            crop = generate_random_crop(imgs1, 56)
            self.patch_prediction_model.visualize(imgs1, imgs2, crop)
    
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
        num_itr = 11000
        # for _ in dataloader:
        #     num_itr += 1
        max_itr = self.num_epochs * num_itr
        self.optimizer = PatchCorrespondenceOptimizer(model_params, init_lr, peak_lr, decay_half_life, warmup_steps, grad_norm_clip, init_weight_decay, peak_weight_decay, max_itr)
        self.optimizer.setup_optimizer()
        self.optimizer.setup_scheduler()
    

    def train_one_epoch(self):
        self.patch_prediction_model.train()
        for i, batch in enumerate(self.dataloader):
            video = batch['video']
            video = video.permute(0, 2, 1, 3, 4)
            imgs1, imgs2 = video[:, 0], video[:, 1]
            imgs1, imgs2 = imgs1.to(self.device), imgs2.to(self.device)
            loss = self.patch_prediction_model.train_step(imgs1, imgs2, crop_size=56)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer.update_lr()
            if i % 10 == 0:
                print("Iteration: {} Loss: {}".format(i, loss.item()))
                wandb.log({"loss": loss.item()})
    
    def train(self):
        for epoch in range(self.num_epochs):
            print("Epoch: {}".format(epoch))
            self.train_one_epoch()
            self.validate(epoch)
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
            # self.logger.log({"val_k=gt_miou": jac})
            print(f"Epoch : {epoch}, eval finished, miou: {jac}")


            


if __name__ == "__main__":
    device = "cuda:3"
    ucf101_path = '/ssdstore/ssalehi/ucf101/data/UCF101'
    clip_durations = 2
    batch_size = 64
    num_workers = 4
    input_size = 224
    logger = wandb.init(project=project_name, group='exp_patch_correspondence', job_type='debug')
    train_transform = trn.Compose(
        [
        ApplyTransformToKey(
            key="video",
            transform=trn.Compose(
                [
                UniformTemporalSubsample(8),
                Lambda(lambda x: x / 255.0),
                Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                RandomShortSideScale(min_size=256, max_size=320),
                trn.Resize((224, 224)),
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
        batch_size=batch_size,  # Adjust the batch size according to your system's capacity
        num_workers=4,  # Adjust the number of workers based on your system's capacity
        pin_memory=True,
    )
    
    vit_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    patch_prediction_model = PatchPredictionModel(224, vit_model, logger=logger)
    optimization_config = {
        'init_lr': 1e-4,
        'peak_lr': 1e-3,
        'decay_half_life': 10000,
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

    patch_prediction_trainer = PatchPredictionTrainer(dataloader, test_dataloader, patch_prediction_model, 100, device, logger)
    patch_prediction_trainer.setup_optimizer(optimization_config)
    patch_prediction_trainer.train()
    patch_prediction_trainer.visualize()


        
