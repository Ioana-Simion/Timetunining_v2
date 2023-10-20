import argparse
from datetime import date
import os
import time
import torch
# from pytorchvideo.data import Ucf101, make_clip_sampler
import torch.nn.functional as F
from clustering import PerDatasetClustering
from data_loader import PascalVOCDataModule, SamplingMode, VideoDataModule
from eval_metrics import PredsmIoU
from evaluator import LinearFinetuneModule
from models import FeatureExtractor, FeatureForwarder, CorrespondenceDetection, SlotAttention
from my_utils import find_optimal_assignment
import wandb
from matplotlib.colors import ListedColormap
from optimizer import TimeTv2Optimizer
import torchvision.transforms as trn

from image_transformations import Compose, Resize
import video_transformations
import numpy as np
import random


project_name = "TimeTuning_v2"
## generate ListeColorMap of distinct colors

## what are the colors for red, blue, green, brown, yello, orange, purple, white, black, maroon, olive, teal, navy, gray, silver
## Fill the ListedColormap with the colors above

cmap = ListedColormap(['#FF0000', '#0000FF', '#008000', '#A52A2A', '#FFFF00', '#FFA500', '#800080', '#FFFFFF', '#000000', '#800000', '#808000', '#008080', '#000080', '#808080', '#C0C0C0'])


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


class TimeTuning_SA(torch.nn.Module):
    def __init__(self, input_size, vit_model, slot_numer=5, num_prototypes=200, topk=5, context_frames=6, context_window=6, logger=None):
        super(TimeTuning_SA, self).__init__()
        self.input_size = input_size
        self.eval_spatial_resolution = input_size // 16
        self.feature_extractor = FeatureExtractor(vit_model, eval_spatial_resolution=self.eval_spatial_resolution, d_model=384)
        self.FF = FeatureForwarder(self.eval_spatial_resolution, context_frames, context_window, topk=topk, feature_head=None)
        self.logger = logger
        self.num_prototypes = num_prototypes
        self.CorDet = CorrespondenceDetection(window_szie=context_window // 2)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.mlp_head = torch.nn.Sequential(
            torch.nn.Linear(self.feature_extractor.d_model, 1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, 256),
        )
        # self.lc = torch.nn.Linear(self.feature_extractor.d_model, self.eval_spatial_resolution ** 2)
        self.feature_extractor.freeze_feature_extractor(["blocks.11", "blocks.10"])
        prototype_init = torch.randn((num_prototypes, 384))
        prototype_init =  F.normalize(prototype_init, dim=-1, p=2)  
        self.prototypes = torch.nn.Parameter(prototype_init)
        self.slot_numer = slot_numer
        self.slot_attention = SlotAttention(slot_numer, self.feature_extractor.d_model)
    

    def normalize_prototypes(self):
        with torch.no_grad():
            w = self.prototypes.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.prototypes.copy_(w)
            



    def train_step(self, datum, crop_list, bboxs):
        self.normalize_prototypes()
        bs, nf, c, h, w = datum.shape
        # denormalized_video = denormalize_video(datum)
        dataset_features, _ = self.feature_extractor.forward_features(datum.flatten(0, 1))
        _, np, dim = dataset_features.shape
        dataset_features = dataset_features.reshape(bs, nf, np, dim)
        first_frame_features = dataset_features[:, 0, :, :]
        last_frame_features = dataset_features[:, -1, :, :]
        first_frames_slots = self.slot_attention(first_frame_features)
        last_frames_slots = self.slot_attention(last_frame_features)
        normalized_frist_frames_slots = F.normalize(first_frames_slots, dim=-1, p=2)
        normalized_last_frames_slots = F.normalize(last_frames_slots, dim=-1, p=2)
        bs, ns, d = normalized_frist_frames_slots.shape
        first_slot_scores = torch.einsum('bd,nd->bn', normalized_frist_frames_slots.view(bs * ns, -1) , self.prototypes)
        first_slot_q = find_optimal_assignment(first_slot_scores, 0.05, 10).reshape(bs, ns, -1)
        last_slot_scores = torch.einsum('bd,nd->bn', normalized_last_frames_slots.view(bs * ns, -1) , self.prototypes)
        last_slot_q = find_optimal_assignment(last_slot_scores, 0.05, 10).reshape(bs, ns, -1)
        first_slot_scores = first_slot_scores.reshape(bs, ns, -1)
        last_slot_scores = last_slot_scores.reshape(bs, ns, -1)
        b_propagated_scores_group = []
        f_propagated_scores_group = []
        for i in range(bs):
            similarity = torch.einsum('ld,kd->lk', normalized_frist_frames_slots[i], normalized_last_frames_slots[i])
            similarity = similarity / 0.03
            similarity = similarity.softmax(dim=1)
            first_slot_scores_i = first_slot_scores[i]
            last_slot_scores_i = last_slot_scores[i]
            f_propagated_first_slot_scores_i = torch.einsum('lk,ln->ln', similarity.T, first_slot_scores_i)
            b_propagated_last_slot_scores_i = torch.einsum('lk,ln->ln', similarity, last_slot_scores_i)
            b_propagated_scores_group.append(b_propagated_last_slot_scores_i)
            f_propagated_scores_group.append(f_propagated_first_slot_scores_i)
        
        b_propagated_scores = torch.stack(b_propagated_scores_group, dim=0)
        f_propagated_scores = torch.stack(f_propagated_scores_group, dim=0)

        loss = self.criterion(f_propagated_scores / 0.1, last_slot_q.argmax(dim=1).long()) + self.criterion(b_propagated_scores / 0.1, first_slot_q.argmax(dim=1).long())
        return loss


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
        SA_params = self.get_params_dict(self.slot_attention,exclude_decay=True, lr=1e-4)
        prototypes_params = [{'params': self.prototypes, 'lr': 1e-4}]
        all_params = feature_extractor_params + mlp_head_params + prototypes_params + SA_params
        return all_params



    def validate_step(self, img):
        self.feature_extractor.eval()
        with torch.no_grad():
            spatial_features, _ = self.feature_extractor.forward_features(img)  # (B, np, dim)
        return spatial_features

    def save(self, path):
        torch.save(self.state_dict(), path)


        

class TimeTuning_SA_Trainer():
    def __init__(self, data_module, test_dataloader, time_tuning_model, num_epochs, device, logger):
        self.dataloader = data_module.data_loader
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
            batch_crop_list, label, annotations = batch
            global_crops_1 = batch_crop_list[0]
            annotations = annotations.squeeze(1)
            datum = global_crops_1
            annotations = annotations.squeeze(1)
            datum = datum.squeeze(1)
            datum = datum.to(self.device)
            for j, crop in enumerate(batch_crop_list):
                batch_crop_list[j] = crop.to(self.device)
            loss = self.time_tuning_model.train_step(datum, batch_crop_list[1:], label["bbox"][:, 1:, ])
            total_loss = loss
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.optimizer.update_lr()
            epoch_loss += total_loss.item()
            print("Iteration: {} Loss: {}".format(i, total_loss.item()))
            self.logger.log({"clustering_loss": loss.item()})
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
        self.time_tuning_model.eval()
        with torch.no_grad():
            metric = PredsmIoU(21, 21)
            # spatial_feature_dim = self.model.get_dino_feature_spatial_dim()
            spatial_feature_dim = 50
            clustering_method = PerDatasetClustering(spatial_feature_dim, 21)
            feature_spatial_resolution = self.time_tuning_model.feature_extractor.eval_spatial_resolution
            feature_group = []
            targets = []
            for i, (x, y) in enumerate(self.test_dataloader):
                target = (y * 255).long()
                img = x.to(self.device)
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
    num_crops = args.num_crops
    num_prototypes = args.num_prototypes
    topk = args.topk
    context_frames = args.context_frames
    context_window = args.context_window
    sampling = args.sampling
    regular_step = args.regular_step
    num_clip_frames = args.num_clip_frames
    num_clips = args.num_clips

    config = vars(args)
    ## make a string of today's date
    today = date.today()
    d1 = today.strftime("%d_%m_%Y")
    logger = wandb.init(project=project_name, mode="disabled", group=d1, job_type='debug_clustering_ytvos', config=config)
    # rand_color_jitter = video_transformations.RandomApply([video_transformations.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.8)
    # data_transform_list = [rand_color_jitter, video_transformations.RandomGrayscale(), video_transformations.RandomGaussianBlur()]
    # data_transform = video_transformations.Compose(data_transform_list)
    # video_transform_list = [video_transformations.Resize(224), video_transformations.RandomResizedCrop((224, 224)), video_transformations.RandomHorizontalFlip(), video_transformations.ClipToTensor(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])] #video_transformations.RandomResizedCrop((224, 224))
    # video_transform = video_transformations.Compose(video_transform_list)
    # num_clips = 1
    # num_clip_frames = 4
    # regular_step = 1
    # transformations_dict = {"data_transforms": data_transform, "target_transforms": None, "shared_transforms": video_transform}
    video_transform_list = [video_transformations.RandomResizedCrop((224, 224)), video_transformations.ClipToTensor()] #video_transformations.RandomResizedCrop((224, 224))
    target_transform = video_transformations.Compose(video_transform_list)
    video_transform = video_transformations.TimeTTransform([224, 96], [1, num_crops], [0.35, 0.25], [1., 0.4], 1, 0.01, 1)
    world_size = 1
    transformations_dict = {"data_transforms": video_transform, "target_transforms": target_transform, "shared_transforms": None}
    prefix = "/var/scratch/ssalehid/ytvos/"
    data_path = os.path.join(prefix, "train1//JPEGImages/")
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

    vit_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    patch_prediction_model = TimeTuning_SA(224, vit_model, num_prototypes=num_prototypes, topk=topk, context_frames=context_frames, context_window=context_window, logger=logger)
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
    patch_prediction_trainer = TimeTuning_SA_Trainer(video_data_module, test_dataloader, patch_prediction_model, num_epochs, device, logger)
    patch_prediction_trainer.setup_optimizer(optimization_config)
    patch_prediction_trainer.train()

    # patch_prediction_trainer.visualize()


            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--ucf101_path', type=str, default="/ssdstore/ssalehi/ucf101/data/UCF101")
    parser.add_argument('--clip_durations', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--num_epochs', type=int, default=800)
    parser.add_argument('--crop_size', type=int, default=64)
    parser.add_argument('--crop_scale_tupple', type=tuple, default=(0.3, 1))
    parser.add_argument('--masking_ratio', type=float, default=1)
    parser.add_argument('--same_frame_query_ref', type=bool, default=False)
    parser.add_argument("--num_crops", type=int, default=4)
    parser.add_argument("--num_prototypes", type=int, default=200)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--context_frames", type=int, default=4)
    parser.add_argument("--context_window", type=int, default=6)
    parser.add_argument("--sampling", type=str, default="dense")
    parser.add_argument("--regular_step", type=int, default=25)
    parser.add_argument("--num_clip_frames", type=int, default=4)
    parser.add_argument("--num_clips", type=int, default=1)
    parser.add_argument("--explaination", type=str, default="Only clustering loss. All_frame dataset")
    args = parser.parse_args()
    run(args)