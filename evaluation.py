import torch
from models import FeatureExtractor
from data_loader import PascalVOCDataModule
import torchvision.transforms as trn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import math
import scann
import time
from eval_metrics import PredsmIoU
import os
import numpy as np

class HummingbirdEvaluation():
    def __init__(self, feature_extractor, dataset_module, num_neighbour, augmentation_epoch, memory_size, device):
        self.feature_extractor = feature_extractor
        self.dataset_module = dataset_module
        self.device = device
        self.augmentation_epoch = augmentation_epoch
        self.memory_size = memory_size
        self.num_neighbour = num_neighbour
        self.feature_extractor.eval()
        self.feature_extractor = feature_extractor.to(self.device)
        self.num_sampled_features = self.memory_size // (self.dataset_module.get_train_dataset_size() * self.augmentation_epoch)
        self.feature_memory, self.label_memory = self.create_memory()
        self.feature_memory = self.feature_memory.to(self.device)
        self.label_memory = self.label_memory.to(self.device)
        self.save_memory()
        self.NN_algorithm = scann.scann_ops_pybind.builder(self.feature_memory.detach().cpu().numpy(), num_neighbour, "dot_product").tree(
    num_leaves=512, num_leaves_to_search=32, training_sample_size=self.feature_memory.size(0)).score_ah(
    2, anisotropic_quantization_threshold=0.2).reorder(120).build()

    def create_memory(self):

        if os.path.isfile("temp/feature_memory.pt") and os.path.isfile("temp/label_memory.pt"):
            feature_memory = torch.load("temp/feature_memory.pt")
            label_memory = torch.load("temp/label_memory.pt")
            return feature_memory, label_memory
        
        memory = []
        label_memory = []
        train_loader = self.dataset_module.get_train_dataloader()
        eval_spatial_resolution = self.feature_extractor.eval_spatial_resolution
        with torch.no_grad():
            for j in range(self.augmentation_epoch):
                print(f"augmentation epoch {j} has started at {time.ctime()}")
                for i, (x, y) in enumerate(train_loader):
                    print(f"batch {i} has been read at {time.ctime()}")
                    x = x.to(self.device)
                    y = y.to(self.device)
                    y = (y * 255).long()
                    print(f"batch {i} has been moved to {self.device} at {time.ctime()}")
                    features, _ = self.feature_extractor.get_intermediate_layer_feats(x)
                    print(f"batch {i} sampling process has been started at {time.ctime()}")
                    input_size = x.shape[-1]
                    patch_size = input_size // eval_spatial_resolution
                    pathified_gts = self.patchify_gt(y, patch_size) ## (bs, spatial_resolution, spatial_resolution, c*patch_size*patch_size)
                    sampled_features, sampled_indices = self.sample_features(features, pathified_gts)
                    normalized_sampled_features = sampled_features / torch.norm(sampled_features, dim=1, keepdim=True)
                    # self.overlay_sampled_locations_on_gt(y, sampled_indices)
                    label = F.interpolate(y.float(), size=(eval_spatial_resolution, eval_spatial_resolution), mode="nearest").long()
                    label = label.flatten(1)
                    ## select the labels of the sampled features
                    sampled_indices = sampled_indices.to(self.device)
                    label_hat = label.gather(1, sampled_indices)
                    label_memory.append(label_hat)
                    memory.append(normalized_sampled_features)
                    print(f"batch {i} has been processed at {time.ctime()}")
        memory = torch.cat(memory)
        label_memory = torch.cat (label_memory)
        memory = memory.flatten(0, 1)
        label_memory = label_memory.flatten(0, 1)
        return memory, label_memory

    def save_memory(self):
        torch.save(self.feature_memory, "temp/feature_memory.pt")
        torch.save(self.label_memory, "temp/label_memory.pt")

    def sample_features(self, features, pathified_gts):
        sampled_features = []
        sampled_indices = []
        for k, gt in enumerate(pathified_gts):
            class_frequency = self.get_class_frequency(gt)
            patch_scores = self.get_patch_scores(gt, class_frequency)
            patch_scores = patch_scores.flatten()
            zero_score_idx = torch.where(patch_scores == 0)
            # assert zero_score_idx[0].size(0) != 0 ## for pascal every patch should belong to one class
            none_zero_score_idx = torch.where(patch_scores != 0)
            patch_scores[zero_score_idx] = 1e6
            ## sample uniform distribution with the size none_zero_score_idx
            uniform_x = torch.rand(none_zero_score_idx[0].size(0))
            patch_scores[none_zero_score_idx] *= uniform_x
            feature = features[k]
            ### select the least num_sampled_features score idndices
            _, indices = torch.topk(patch_scores, self.num_sampled_features, largest=False)
            sampled_indices.append(indices)
            samples = feature[indices]
            sampled_features.append(samples)
        sampled_features = torch.stack(sampled_features)
        sampled_indices = torch.stack(sampled_indices)
        return sampled_features, sampled_indices

    def get_class_frequency(self, gt):
        class_frequency = torch.zeros((self.dataset_module.get_num_classes()))
        for i in range(self.dataset_module.get_num_classes()):
            class_existence = torch.zeros((gt.shape[0], gt.shape[1]))
            for j in range(gt.shape[0]):
                for k in range(gt.shape[1]):
                    if torch.sum(gt[j, k] == i) > 0:
                        class_existence[j, k] = 1
            class_frequency[i] = torch.sum(class_existence)
        return class_frequency
    

    def get_patch_scores(self, gt, class_frequency):
        patch_scores = torch.zeros((gt.shape[0], gt.shape[1]))
        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                for k in range(class_frequency.shape[0]):
                    if torch.sum(gt[i, j] == k) > 0:
                        patch_scores[i, j] += class_frequency[k]
        return patch_scores
    
    
    def patchify_gt(self, gt, patch_size):
        bs, c, h, w = gt.shape
        gt = gt.reshape(bs, c, h//patch_size, patch_size, w//patch_size, patch_size)
        gt = gt.permute(0, 2, 4, 1, 3, 5)
        gt = gt.reshape(bs, h//patch_size, w//patch_size, c*patch_size*patch_size)
        return gt
    

    def overlay_sampled_locations_on_gt(self, gts, sampled_indices):
        """
        This function overlays the sampled locations on the ground truth 
        and saves the figure in temp folder. It is used to check if the
        sampling is done correctly. For better visualization, turn off the
        uniform sampling when calling the sample_features function.

        Args:
            gts (torch.Tensor): ground truth tensor of shape (bs, c, h, w)
            sampled_indices (torch.Tensor): sampled indices of shape (bs, num_sampled_features)
        """
        maps = torch.zeros_like(gts)
        ## downsample the map to eval_spatial_resolution
        maps = F.interpolate(maps.float(), size=(self.feature_extractor.eval_spatial_resolution, self.feature_extractor.eval_spatial_resolution), mode="nearest").long()
        for i in range(maps.shape[0]):
            map = maps[i]
            sampled_idx = sampled_indices[i]
            map = map.flatten()
            map[sampled_idx] = 1
            map = map.reshape(1, self.feature_extractor.eval_spatial_resolution, self.feature_extractor.eval_spatial_resolution)
            maps[i] = map
        
        maps = F.interpolate(maps.float(), size=(gts.shape[2], gts.shape[3]), mode="nearest").long()
        ## save figures of maps and gts together
        for i in range(maps.shape[0]):
            map = maps[i]
            gt = gts[i]
            plt.imshow(gt.detach().cpu().numpy().transpose(1, 2, 0))
            plt.imshow(map.detach().cpu().numpy().transpose(1, 2, 0), alpha=0.1)
            plt.savefig(f"temp/map_{i}.png")
            plt.close()

    def recall(self, x):
        query_features, _ = self.feature_extractor.get_intermediate_layer_feats(x)


    
    def cross_attention(self, q, k, v, beta=0.02):
        """
        Args: 
            q (torch.Tensor): query tensor of shape (bs, num_patches, d_k)
            k (torch.Tensor): key tensor of shape (bs, num_patches,  num_sampled_features, d_k)
            v (torch.Tensor): value tensor of shape (bs, num_patches, num_sampled_features, label_dim)
        """
        d_k = q.size(-1)
        q = q / torch.norm(q, dim=-1, keepdim=True)
        k = k / torch.norm(k, dim=-1, keepdim=True)
        q = q.unsqueeze(2) ## (bs, num_patches, 1, d_k)
        attn = torch.einsum("bnld,bnmd->bnlm", q, k) / beta ## (bs, num_patches, num_sampled_features)
        attn = attn.squeeze(2)
        attn = F.softmax(attn, dim=-1)
        attn = attn.unsqueeze(-1)
        label_hat = torch.einsum("blms,blmk->blsk", attn, v)
        label_hat = label_hat.squeeze(-2)
        return label_hat
    
    def find_nearest_key_to_query(self, q):
        bs, num_patches, d_k = q.shape
        reshaped_q = q.reshape(bs*num_patches, d_k).detach().cpu().numpy()
        neighbors, distances = self.NN_algorithm.search_batched(reshaped_q)
        neighbors = neighbors.astype(np.int64)
        neighbors = torch.from_numpy(neighbors).to(self.device)
        neighbors = neighbors.flatten()
        key_features = self.feature_memory[neighbors]
        key_features = key_features.reshape(bs, num_patches, self.num_neighbour, -1)
        key_labels = self.label_memory[neighbors]
        key_labels = key_labels.reshape(bs, num_patches, self.num_neighbour)
        ## convert key_labels to one hot
        key_labels = F.one_hot(key_labels, num_classes=self.dataset_module.get_num_classes()).float()
        return key_features, key_labels

    def incontext_evaluation(self):
        metric = PredsmIoU(21, 21)
        val_loader = self.dataset_module.get_val_dataloader()
        eval_spatial_resolution = self.feature_extractor.eval_spatial_resolution
        lebel_hats = []
        lables = []
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                y = (y * 255).long()
                features, _ = self.feature_extractor.get_intermediate_layer_feats(x)
                key_features, key_labels = self.find_nearest_key_to_query(features)
                label_hat =  self.cross_attention(features, key_features, key_labels)
                cluster_map = label_hat.argmax(dim=-1)
                cluster_map = cluster_map.reshape(x.shape[0], eval_spatial_resolution, eval_spatial_resolution).unsqueeze(1)
                lebel_hats.append(cluster_map)
                resized_labels =  F.interpolate(y.float(), size=(eval_spatial_resolution, eval_spatial_resolution), mode="nearest").long()
                lables.append(resized_labels)
        
        lables = torch.cat(lables)
        label_hats = torch.cat(lebel_hats)
        valid_idx = lables != 255
        valid_target = lables[valid_idx]
        valid_cluster_maps = label_hats[valid_idx]
        metric.update(valid_target, valid_cluster_maps)
        jac, tp, fp, fn, reordered_preds, matched_bg_clusters = metric.compute(is_global_zero=True)
        print(f"eval finished, miou: {jac}")

                

  


if __name__ == "__main__":
    vit_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    feature_extractor = FeatureExtractor(vit_model)
    image_train_transform = trn.Compose([trn.Resize((224, 224)), trn.ToTensor(), trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])])
    target_train_transform = trn.Compose([trn.Resize((224, 224), interpolation=trn.InterpolationMode.NEAREST), trn.ToTensor()])
    train_transforms = {"img": image_train_transform, "target": target_train_transform}
    dataset = PascalVOCDataModule(batch_size=128, train_transform=train_transforms, val_transform=train_transforms, test_transform=train_transforms)
    dataset.setup()
    evaluator = HummingbirdEvaluation(feature_extractor, dataset, num_neighbour=10, augmentation_epoch=1, memory_size=200000, device="cuda:2")
    evaluator.incontext_evaluation()