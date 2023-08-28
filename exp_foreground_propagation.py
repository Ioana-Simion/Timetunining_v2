import argparse
from collections import defaultdict
import os
import time
import numpy as np
import torch
import wandb
from clustering import PerDatasetClustering
from data_loader import SamplingMode, VideoDataModule
from eval_metrics import PredsmIoU
from models import FeatureForwarder, FeatureExtractor
from optimizer import PatchCorrespondenceOptimizer
import video_transformations
import faiss
from my_utils import process_attentions
from PIL import Image
import torch.nn.functional as F



project_name = "TimeTuning_v2"




class MaskExtractor():

    """	
    This class takes in a feature extractor and a clustering algorithm and returns the masks for each object in the video.
    The vit_feature_extractor is a feature extractor that takes in a video and returns the features for each frame.
    The clustering algorithm takes in the features and returns the cluster maps for each frame.
    The clustering algorithm is done by stacking the features for each frame and clustering them.
    """

    def __init__(self, feature_extractor, num_clusters):
        self.num_clusters = num_clusters
        self.feature_extractor = feature_extractor





class ForegroundEnhancement(torch.nn.Module):
    def __init__(self, feature_extractor, feature_forwarder, num_clusters=500, num_prototypes=2):
        super(ForegroundEnhancement, self).__init__()
        self.feature_extractor = feature_extractor
        self.feature_forwarder = feature_forwarder
        self.feature_extractor.eval()
        self.num_clusters = num_clusters
        self.spatial_resolution = self.feature_extractor.eval_spatial_resolution
        self.kmeans = faiss.Kmeans(self.feature_extractor.d_model, self.num_clusters, niter=20, verbose=True, gpu=True)
        self.num_prototypes = num_prototypes
        self.feature_extractor.freeze_feature_extractor(["blocks.11", "blocks.10"])
        self.lc = torch.nn.Linear(self.feature_extractor.d_model, 64)
        self.prototypes = torch.nn.Parameter(torch.randn(num_prototypes, 64))
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def extract_sailiency_masks(self, imgs):
        bs, nf, c, h, w = imgs.shape
        imgs = imgs.reshape(bs*nf, c, h, w)
        attentions = self.feature_extractor.forward_features(imgs)
        normalized_cls_attention = process_attentions(attentions[:, :, 0, 1:], self.spatial_resolution)
        return normalized_cls_attention
    

    def return_segmented_input(self, features, masks=None):
        bs_nf, np, d = features.shape
        features = features.reshape(-1, d)
        if masks is not None:
            masks = F.interpolate(masks.unsqueeze(1).float(), size=self.spatial_resolution, mode="nearest").long()
            masks = masks.reshape(-1, 1)
            features = features * masks
        feature_numpy = features.cpu().detach().numpy()
        self.kmeans.train(feature_numpy)
        D, I = self.kmeans.index.search(feature_numpy, 1)
        cluster_maps = torch.from_numpy(I.reshape(bs_nf, np)).to(self.device)
        return cluster_maps


    def extract_soft_fg(self, threshold, features, attentions):
        bs, _, _ = features.shape
        attention = process_attentions(attentions[:, :, 0, 1:], self.feature_extractor.eval_spatial_resolution).squeeze()
        cluster_maps  = self.return_segmented_input(features)
        cluster_maps = cluster_maps.reshape(bs, self.spatial_resolution, self.spatial_resolution)
        cluster_maps = cluster_maps.squeeze()
        precision = self.get_cluster_precs(cluster_maps, attention, self.num_clusters)
        attn_mask_soft = self.make_post_matching_maps(cluster_maps, threshold, precision)
        return attn_mask_soft


    def set_num_clusters(self, num_clusters):
        self.num_clusters = num_clusters
        self.kmeans = faiss.Kmeans(self.feature_dimension, self.num_clusters, niter=20, verbose=True, gpu=True)

    def make_post_matching_maps(self, cluster_maps, threshold, cluster_precs):
        start_idx = np.where((np.sort(cluster_precs) >= threshold) == True)[0][0]
        fg_ids = np.argsort(cluster_precs)[start_idx:]
        attn_mask_soft = torch.zeros_like(cluster_maps)
        for i in fg_ids:
            attn_mask_soft[cluster_maps == i] = 1
        return attn_mask_soft

    def get_cluster_precs(self, cluster, mask, k):
        # Calculate attention foreground precision for each cluster id.
        # Note this doesn't use any gt but rather takes the ViT attention as noisy ground-truth for foreground.
        assert cluster.size(0) == mask.size(0)
        cluster_id_to_oc_count = defaultdict(int)
        cluster_id_to_cum_jac = defaultdict(float)
        for img_id in range(cluster.size(0)):
            img_attn = mask[img_id].flatten()
            img_clus = cluster[img_id].flatten()
            for cluster_id in torch.unique(img_clus):
                tmp_attn = (img_attn == 1)
                tmp_clust = (img_clus == cluster_id)
                tp = torch.sum(tmp_attn & tmp_clust).item()
                fp = torch.sum(~tmp_attn & tmp_clust).item()
                prec = float(tp) / max(float(tp + fp), 1e-8)  # Calculate precision
                cluster_id_to_oc_count[cluster_id.item()] += 1
                cluster_id_to_cum_jac[cluster_id.item()] += prec
        assert len(cluster_id_to_oc_count.keys()) == k and len(cluster_id_to_cum_jac.keys()) == k
        # Calculate average precision values
        precs = []
        for cluster_id in sorted(cluster_id_to_oc_count.keys()):
            precs.append(cluster_id_to_cum_jac[cluster_id] / cluster_id_to_oc_count[cluster_id])
        return precs

    def train_step(self, video):
        bs, nf, c, h, w = video.shape
        features, attentions = self.feature_extractor.forward_features(video.reshape(bs*nf, c, h, w))
        features = features.reshape(bs, nf, -1)
        attentions = attentions.reshape(bs, nf, -1)
        first_frame_attn = self.extract_soft_fg(self, 0.3, features[:, 0], attentions[:, 0])
        one_hot_segmentation = torch.nn.functional.one_hot(first_frame_attn.long(), self.num_prototypes).permute(2, 0, 1).float()
        features = features.reshape(bs, nf, self.spatial_resolution * self.spatial_resolution, -1)
        batch_loss = 0
        for i, feature in enumerate(features):
            prediction = self.feature_forwarder(feature, one_hot_segmentation[i])
            last_frame_prediction = prediction[-1]
            _, fg_mask = torch.max(last_frame_prediction, dim=0)
            scores = self.get_similarity_scores(feature[:, -1], self.prototypes)
            scores = scores.reshape(bs, self.spatial_resolution, self.spatial_resolution, self.num_prototypes)
            scores = scores.permute(0, 3, 1, 2)
            loss = self.criterion(scores, fg_mask)
            batch_loss += loss
        batch_loss /= bs
        return batch_loss
    

    def get_similarity_scores(self, features, prototypes):
        bs, np, d = features.shape
        features = features.reshape(bs*np, d)
        prototypes = prototypes.reshape(self.num_prototypes, d)
        normalized_features = F.normalize(features, dim=1)
        normalized_prototypes = F.normalize(prototypes, dim=1)
        similarity_scores = torch.mm(features, prototypes.t())
        similarity_scores = similarity_scores.reshape(bs, np, self.num_prototypes)
        return similarity_scores


    def get_optimization_params(self):
        return [
            {"params": self.feature_extractor.parameters(), "lr": 1e-5},
            {"params": self.prototypes.parameters(), "lr": 1e-4},
            {"params": self.lc.parameters(), "lr": 1e-4},
        ]
    


class ForegroundEnhancementModule():
    def __init__(self, dataloader, test_dataloader, foreground_enhancement_model, num_epochs, device, logger):
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        self.foreground_enhancement_model = foreground_enhancement_model
        self.device = device
        self.foreground_enhancement_model = self.foreground_enhancement_model.to(self.device)
        self.optimizer = None
        self.num_epochs = num_epochs
        self.logger = logger
        self.logger.watch(foreground_enhancement_model, log="all", log_freq=10)
    
    
    def setup_optimizer(self, optimization_config):
        model_params = self.foreground_enhancement_model.get_optimization_params()
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
        self.foreground_enhancement_model.train()
        epoch_loss = 0
        before_loading_time = time.time()
        for i, batch in enumerate(self.dataloader):
            after_loading_time = time.time()
            print("Loading Time: {}".format(after_loading_time - before_loading_time))
            datum, annotations = batch
            annotations = annotations.squeeze(1)
            datum = datum.squeeze(1)
            loss = self.foreground_enhancement_model.train_step(datum)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer.update_lr()
            epoch_loss += loss.item()
            print("Iteration: {} Loss: {}".format(i, loss.item()))
            self.logger.log({"loss": loss.item()})
            lr = self.optimizer.get_lr()
            self.logger.log({"lr": lr})
            before_loading_time = time.time()
        epoch_loss /= (i + 1)
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
    






def run(args):
    device = args.device
    batch_size = args.batch_size
    num_workers = args.num_workers
    input_size = args.input_size
    num_clips = args.num_clips
    num_workers = args.num_workers
    num_clip_frames = args.num_clip_frames
    regular_step = args.regular_step
    context_frames = args.context_frames
    context_window = args.context_window
    topk = args.topk
    uvos_flag = args.uvos_flag
    precision_based = args.precision_based
    many_to_one = args.many_to_one

    logger = wandb.init(project=project_name, group='exp_patch_correspondence', job_type='debug')
    rand_color_jitter = video_transformations.RandomApply([video_transformations.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.8)
    data_transform_list = [rand_color_jitter, video_transformations.RandomGrayscale(), video_transformations.RandomGaussianBlur()]
    data_transform = video_transformations.Compose(data_transform_list)
    video_transform_list = [video_transformations.Resize((input_size, input_size), 'bilinear'), video_transformations.ClipToTensor(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])] #video_transformations.RandomResizedCrop((224, 224))
    video_transform = video_transformations.Compose(video_transform_list)
    transformations_dict = {"data_transforms": None, "target_transforms": None, "shared_transforms": video_transform}
    prefix = "/ssdstore/ssalehi/dataset"
    data_path = os.path.join(prefix, "davis_2021/davis_data/JPEGImages/")
    annotation_path = os.path.join(prefix, "davis_2021/DAVIS/Annotations/")
    # meta_file_path = os.path.join(prefix, "train1/meta.json")
    meta_file_path = None
    path_dict = {"class_directory": data_path, "annotation_directory": annotation_path, "meta_file_path": meta_file_path}
    sampling_mode = SamplingMode.Full
    video_data_module = VideoDataModule("davis", path_dict, num_clips, num_clip_frames, sampling_mode, regular_step, batch_size, num_workers)
    video_data_module.setup(transformations_dict)
    data_loader = video_data_module.get_data_loader()
    vit_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    feature_extractor = FeatureExtractor(vit_model, 14, d_model=384)
    FF = FeatureForwarder(feature_extractor.eval_spatial_resolution, context_frames, context_window, topk, feature_head=None)
    FF = FF.to(device)
    FF.eval()
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    predictions = []
    scores = [] 
    for batch_idx, batch in enumerate(data_loader):
        inputs, annotations = batch
        inputs = inputs.to(device).squeeze()
        annotations = annotations.to(device).squeeze()
        if uvos_flag:
            idx = annotations > 0
            annotations[idx] = 1
        features, _ = feature_extractor.forward_features(inputs)
        first_frame_segmentation = annotations[0]
        n_dims = int(first_frame_segmentation.max()+ 1)
        one_hot_segmentation = torch.nn.functional.one_hot(first_frame_segmentation.long(), n_dims).permute(2, 0, 1).float()
        prediction = FF(features, one_hot_segmentation)
        prediction = torch.stack(prediction, dim=0)
        prediction = torch.nn.functional.interpolate(prediction, size=(inputs.size(-2), inputs.size(-1)), mode="nearest")
        _, prediction = torch.max(prediction, dim=1)
        prediction = adjust_max(prediction)
        annotations = adjust_max(annotations)
        num_classes = len(torch.unique(annotations))
        predictions.append(prediction)
        predsmIoU = PredsmIoU(num_classes, num_classes)
        predsmIoU.update(prediction.flatten(), annotations[1:].flatten())
        score, tp, fp, fn, reordered_preds, matched_bg_clusters = predsmIoU.compute(True, many_to_one, precision_based=precision_based)
        scores.append(score)
    
    print("Mean IoU: {}".format(torch.mean(torch.stack(scores))))

def adjust_max(input):
    input = input
    unique = torch.unique(input)
    for i in range(len(unique)):
        input[input == unique[i]] = i
    return input


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--device", type=str, default="cuda:4")
    args.add_argument("--batch_size", type=int, default=1)
    args.add_argument("--num_workers", type=int, default=8)
    args.add_argument("--input_size", type=int, default=224)
    args.add_argument("--num_clips", type=int, default=1)
    args.add_argument("--num_clip_frames", type=int, default=4)
    args.add_argument("--regular_step", type=int, default=1)
    args.add_argument("--context_frames", type=int, default=4)
    args.add_argument("--context_window", type=int, default=2)
    args.add_argument("--topk", type=int, default=4)
    args.add_argument("--uvos_flag", type=bool, default=False)
    args.add_argument("--precision_based", type=bool, default=False)
    args.add_argument("--many_to_one", type=bool, default=False)
    args = args.parse_args()
    run(args)