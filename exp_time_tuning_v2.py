
import argparse
from datetime import date
import os
import time
import torch
# from pytorchvideo.data import Ucf101, make_clip_sampler
import torch.nn.functional as F
from clustering import PerDatasetClustering
from data_loader import PascalVOCDataModule, SamplingMode, VideoDataModule, SPairDataset, CLASS_IDS
from eval_metrics import PredsmIoU
from evaluator import LinearFinetuneModule, KeypointMatchingModule
from models import FeatureExtractor, FeatureForwarder
from my_utils import find_optimal_assignment, denormalize_video
import wandb
from matplotlib.colors import ListedColormap
from optimizer import TimeTv2Optimizer
import torchvision.transforms as trn

from image_transformations import Compose, Resize
import video_transformations
import numpy as np
import random
import copy


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


class TimeTuningV2(torch.nn.Module):
    def __init__(self, input_size, vit_model, num_prototypes=200, topk=5, context_frames=6, context_window=6, logger=None, model_type='dino', training_set = 'ytvos', use_neco_loss=False):
        super(TimeTuningV2, self).__init__()
        self.input_size = input_size
        if model_type == 'dino':
            self.eval_spatial_resolution = input_size // 16
        elif model_type in ['dinov2', 'registers']:
            self.eval_spatial_resolution = input_size // 14
        self.feature_extractor = FeatureExtractor(
            vit_model,
            eval_spatial_resolution=self.eval_spatial_resolution,
            d_model=384,
            model_type=model_type,
            #num_registers=4 if model_type == "registers" else 0,
            num_registers=0,
        )
        self.FF = FeatureForwarder(self.eval_spatial_resolution, context_frames, context_window, topk=topk, feature_head=None)
        self.logger = logger
        self.num_prototypes = num_prototypes
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
        self.feature_extractor.freeze_feature_extractor(["blocks.11", "blocks.10"])
        prototype_init = torch.randn((num_prototypes, 256))
        prototype_init = F.normalize(prototype_init, dim=-1, p=2)
        self.prototypes = torch.nn.Parameter(prototype_init)
        self.model_type = model_type
        self.training_set = training_set
        self.use_neco_loss = use_neco_loss
        self.teacher_model = copy.deepcopy(self.feature_extractor)


    def normalize_prototypes(self):
        with torch.no_grad():
            w = self.prototypes.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.prototypes.copy_(w)

    def compute_knn_perm_mat(self, source_features, target_features, sim_metric='euclidean'):
        """
        Compute KNN permutation matrix between source and target features.

        Args:
            source_features: Tensor of shape (B, H*W, D) - Source features (teacher or student).
            target_features: Tensor of shape (N, H*W, D) - Target features (reference).
            sim_metric: Similarity metric ('euclidean' or 'cosine').

        Returns:
            Permutation matrix of shape (B, H*W, H*W).
        """
        if sim_metric == 'euclidean':
            # Compute pairwise distances
            distances = torch.cdist(source_features, target_features, p=2.0)  # (B, H*W, N*H*W)
        elif sim_metric == 'cosine':
            # Compute cosine similarities
            distances = -torch.einsum('bhd,nhd->bhn', 
                                    F.normalize(source_features, dim=-1), 
                                    F.normalize(target_features, dim=-1))  # (B, H*W, N*H*W)
        else:
            raise ValueError(f"Unsupported similarity metric: {sim_metric}")

        # Create a permutation matrix (one-hot-like)
        perm_matrix = torch.softmax(-distances, dim=-1)  # Convert distances to probabilities
        return perm_matrix


    def compute_knn_subloss(self, teacher_perm, student_perm):
        """
        Computes KNN loss between teacher and student permutation matrices.

        Args:
            teacher_perm: Permutation matrix for teacher features (B, H*W, H*W).
            student_perm: Permutation matrix for student features (B, H*W, H*W).

        Returns:
            KNN loss value.
        """
        return torch.mean(torch.sum(teacher_perm * torch.log(student_perm + 1e-6), dim=-1))  # Stability via +1e-6


    def train_step(self, datum, reference_features=None): 
        """
        Perform a training step, either using CrossEntropy loss or NeCo loss based on the configuration.
        """
        self.normalize_prototypes()

        # Extract student embeddings
        bs, nf, c, h, w = datum.shape
        student_features, _ = self.feature_extractor.forward_features(datum.flatten(0, 1))
        student_features = student_features.view(bs, nf, self.eval_spatial_resolution, self.eval_spatial_resolution, -1)

        if self.use_neco_loss:
            # Extract features
            teacher_frame = datum[:, -1, :, :, :]  # Last frame
            student_frames = datum[:, :-1, :, :, :]  # All other frames
            teacher_features, _ = self.feature_extractor.forward_features(teacher_frame)
            student_features, _ = self.feature_extractor.forward_features(student_frames.flatten(0, 1))
            print(f"Shape of student_features: {student_features.shape}")

            spatial_resolution = int(student_features.shape[1] ** 0.5)  # Assuming square spatial layout
            feature_dim = student_features.shape[-1]  # Feature depth
            student_features = student_features.view(bs, nf - 1, spatial_resolution, spatial_resolution, feature_dim)
            #student_features = student_features.view(datum.size(0), datum.size(1) - 1, -1, -1, -1)
            reference_features = [torch.tensor(feature) for feature in reference_features]
            reference_features = torch.stack(reference_features)
            reference_features = F.normalize(reference_features, dim=-1)
            # Normalize and reshape projected_teacher_features
            projected_teacher_features = F.normalize(self.mlp_head(teacher_features), dim=-1)
            print(f"Shape of projected_teacher_features: {projected_teacher_features.shape}")
            projected_teacher_features = projected_teacher_features.view(-1, projected_teacher_features.shape[-1])
            print(f"Shape of projected_teacher_features after view: {projected_teacher_features.shape}")
            if len(reference_features.shape) > 2:
                reference_features = reference_features.view(-1, reference_features.shape[-1])
            print(f"Shape of reference_features beforer mlp: {reference_features.shape}")
            if reference_features.shape[-1] != projected_teacher_features.shape[-1]:
                reference_features = self.mlp_head(reference_features)
                reference_features = F.normalize(reference_features, dim=-1)  # Shape: (N, 256)
            print(f"Shape of reference_features after mlp: {reference_features.shape}")
            # Compute the first_segmentation_map using find_optimal_assignment
            #projected_teacher_features = F.normalize(self.mlp_head(teacher_features), dim=-1)
            teacher_scores = torch.einsum('bd,nd->bn', projected_teacher_features, reference_features)
            first_segmentation_map = find_optimal_assignment(teacher_scores, epsilon=0.05, sinkhorn_iterations=10)
            first_segmentation_map = first_segmentation_map.reshape(
                bs, spatial_resolution, spatial_resolution, self.num_prototypes
            ).permute(0, 3, 1, 2)

            # Align features
            aligned_teacher_features, aligned_student_features = self.FF.forward_align_features(
                teacher_features, student_features, first_segmentation_map
            )

            # Normalize features
            aligned_teacher_features = F.normalize(aligned_teacher_features, dim=-1)
            aligned_student_features = F.normalize(aligned_student_features, dim=-1)

            # Compute permutation matrices
            perm_matrix_teacher = self.compute_knn_perm_mat(aligned_teacher_features, reference_features)
            perm_matrix_student = self.compute_knn_perm_mat(aligned_student_features, reference_features)

            # Compute KNN loss
            loss = self.compute_knn_subloss(perm_matrix_teacher, perm_matrix_student)

            # Optionally, update teacher using EMA
            self.update_teacher()
        else:
            # Original CrossEntropy Loss Logic
            _, np, dim = student_features.shape
            projected_patch_features = self.mlp_head(student_features)
            projected_dim = projected_patch_features.shape[-1]
            projected_patch_features = projected_patch_features.reshape(-1, projected_dim)
            normalized_projected_features = F.normalize(projected_patch_features, dim=-1, p=2)

            dataset_scores = torch.einsum('bd,nd->bn', normalized_projected_features, self.prototypes)
            dataset_q = find_optimal_assignment(dataset_scores, 0.05, 10)
            dataset_q = dataset_q.reshape(bs, nf, np, self.num_prototypes)
            dataset_scores = dataset_scores.reshape(bs, nf, np, self.num_prototypes)
            dataset_first_frame_q = dataset_q[:, 0, :, :]
            dataset_target_frame_scores = dataset_scores[:, -1, :, :]
            dataset_first_frame_q = dataset_first_frame_q.reshape(
                bs, self.eval_spatial_resolution, self.eval_spatial_resolution, self.num_prototypes
            ).permute(0, 3, 1, 2).float()

            # Temporal propagation
            target_scores_group = []
            q_group = []
            for i, clip_features in enumerate(student_features):
                q = dataset_first_frame_q[i]
                target_frame_scores = dataset_target_frame_scores[i]
                prediction = self.FF.forward(clip_features, q)
                prediction = torch.stack(prediction, dim=0)
                propagated_q = prediction[-1]
                target_frame_scores = target_frame_scores.reshape(
                    self.eval_spatial_resolution, self.eval_spatial_resolution, self.num_prototypes
                ).permute(2, 0, 1).float()
                target_scores_group.append(target_frame_scores)
                q_group.append(propagated_q)

            target_scores = torch.stack(target_scores_group, dim=0)
            propagated_q_group = torch.stack(q_group, dim=0)
            propagated_q_group = propagated_q_group.argmax(dim=1)
            loss = self.criterion(target_scores / 0.1, propagated_q_group.long())

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
        prototypes_params = [{'params': self.prototypes, 'lr': 1e-4}]
        all_params = feature_extractor_params + mlp_head_params + prototypes_params
        return all_params

    def validate_step(self, img):
        self.feature_extractor.eval()
        with torch.no_grad():
            spatial_features, reg = self.feature_extractor.forward_features(img)  # (B, np, dim)
            # if self.model_type ==  "registers":
            #     # Exclude registers during validation
            #     print(f'spatial_features shape before: {spatial_features.shape}')
            #     print(f'reg shape before: {reg.shape}')
            #     spatial_features = spatial_features[:, :-4, :]  # Last 8 are registers
        return spatial_features
    
    def update_teacher(self, momentum=0.999):
        """Update teacher model"""
        for teacher_param, student_param in zip(self.teacher_model.parameters(), self.feature_extractor.parameters()):
            teacher_param.data.mul_(momentum).add_((1 - momentum) * student_param.data)

    def save(self, path):
        torch.save(self.state_dict(), path)

        

class TimeTuningV2Trainer():
    def __init__(self, data_module, test_dataloader, time_tuning_model, num_epochs, device, logger, spair_dataset, spair_val=False, use_neco_loss=False):
        self.dataloader = data_module.data_loader
        self.test_dataloader = test_dataloader
        self.time_tuning_model = time_tuning_model
        self.device = device
        self.time_tuning_model = self.time_tuning_model.to(self.device)
        self.optimizer = None
        self.num_epochs = num_epochs
        self.logger = logger
        self.logger.watch(time_tuning_model, log="all", log_freq=10)
        self.best_miou = 0
        self.best_recall = 0
        self.spair_val = spair_val
        self.use_neco_loss = use_neco_loss
        if self.spair_val:
            # print(f'spair_data_path: {spair_data_path}')
            # spair_dataset = SPairDataset(
            #         root=spair_data_path,
            #         split="test",
            #         use_bbox=False,
            #         image_size=224,
            #         image_mean="imagenet",
            #         class_name=list(CLASS_IDS.keys()),
            #         num_instances=100,
            #         vp_diff=None,
            # )
            # print(f'Length of SPair Dataset: {len(spair_dataset)}')
            self.spair_dataset = spair_dataset
            #eval_model = copy.deepcopy(self.time_tuning_model)
            #eval_model.eval()

            #self.keypoint_matching_module = KeypointMatchingModule(eval_model, spair_dataset, device)

    
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
    
    def gather_references(self, max_references=50):
        """
        Pre-gather reference features for all clips in the dataset.
        
        Args:
            dataloader: PyTorch DataLoader for the dataset.
            model: Feature extractor model (student).
            device: Device for computation (e.g., 'cuda').
            max_references: Maximum number of reference features to collect.
            
        Returns:
            reference_buffer: A list of reference features (Fr).
        """
        reference_buffer = []
        self.time_tuning_model.eval()  # Set the model to evaluation mode
        
        with torch.no_grad():
            for batch in self.dataloader:
                datum, _ = batch
                datum = datum.squeeze(1).to(self.device)
                
                # Extract features from the student model
                print(f"Shape of datum: {datum.shape}")
                features, _ = self.time_tuning_model.feature_extractor.forward_features(datum.flatten(0, 1))
                reference_buffer.extend(features.detach().cpu().numpy())  # Convert to numpy for storage
                
                # Stop if we've gathered enough references
                if len(reference_buffer) >= max_references:
                    break
        
        # Limit to max_references
        reference_buffer = reference_buffer[:max_references]
        print(f"Gathered {len(reference_buffer)} reference features.")
        return reference_buffer


    def train_one_epoch(self, reference_buffer=None):
        self.time_tuning_model.train()
        epoch_loss = 0
        before_loading_time = time.time()
        for i, batch in enumerate(self.dataloader):
            after_loading_time = time.time()
            print("Loading Time: {}".format(after_loading_time - before_loading_time))
            datum, annotations = batch
            datum = datum.squeeze(1).to(self.device)
            if reference_buffer:
                # Sample references from the pre-gathered buffer
                reference_frames = random.sample(reference_buffer, min(len(reference_buffer), 20))  # Use 20 references per batch
            else:
                reference_frames = None
            clustering_loss = self.time_tuning_model.train_step(datum,reference_frames)
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
        print("Epoch Loss: {}".format(epoch_loss))
    
    def train(self):
        for epoch in range(self.num_epochs):
            print("Epoch: {}".format(epoch))
            # if epoch % 1 == 0:
            #     self.validate(epoch)
            if self.spair_val:
                #if epoch % 10 == 0: # 2 only for debuggingt then we do evey 10/20
                self.validate(epoch)
                #else:
                eval_model = copy.deepcopy(self.time_tuning_model)
                eval_model.eval()
                self.time_tuning_model.eval()
                self.keypoint_matching_module = KeypointMatchingModule(eval_model, self.spair_dataset, self.device)
                recall, _ = self.keypoint_matching_module.evaluate_dataset(self.spair_dataset, thresh=0.10)
                self.logger.log({"keypoint_matching_recall": recall})
                print(f"Keypoint Matching Recall at epoch {epoch}: {recall:.2f}%")
                if recall > self.best_recall:
                    self.best_recall = recall
                    checkpoint_dir = "checkpoints"
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    save_path = os.path.join(checkpoint_dir, f"model_best_recall_epoch_{epoch}_{self.time_tuning_model.model_type}_{self.time_tuning_model.training_set}_FIXED_reg2.pth")
                    torch.save(self.time_tuning_model.state_dict(), save_path)
                    print(f"Model saved with best recall: {self.best_recall:.2f}% at epoch {epoch}")
            else:
                self.validate(epoch)
            if self.use_neco_loss:
                self.train_one_epoch(self.gather_references())
            else:
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
            print(f"eval_features shape before reshape: {eval_features.shape}")
            eval_targets = torch.cat(targets, dim=0)
            B, np, dim = eval_features.shape
            eval_features = eval_features.reshape(eval_features.shape[0], feature_spatial_resolution, feature_spatial_resolution, dim)
            eval_features = eval_features.permute(0, 3, 1, 2).contiguous()
            eval_features = F.interpolate(eval_features, size=(val_spatial_resolution, val_spatial_resolution), mode="bilinear")
            eval_features = eval_features.reshape(B, dim, -1).permute(0, 2, 1)
            eval_features = eval_features.detach().cpu().unsqueeze(1)
            print(f"feature_spatial_resolution: {feature_spatial_resolution}, dim: {dim}")
            cluster_maps = clustering_method.cluster(eval_features)
            cluster_maps = cluster_maps.reshape(B, val_spatial_resolution, val_spatial_resolution).unsqueeze(1)
            valid_idx = eval_targets != 255
            valid_target = eval_targets[valid_idx]
            valid_cluster_maps = cluster_maps[valid_idx]
            metric.update(valid_target, valid_cluster_maps)
            jac, tp, fp, fn, reordered_preds, matched_bg_clusters = metric.compute(is_global_zero=True)
            self.logger.log({"val_k=gt_miou": jac})
            # print(f"Epoch : {epoch}, eval finished, miou: {jac}")
            checkpoint_dir = "checkpoints"
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            #threshold = 0.143
            if jac > self.best_miou: #self.best_miou:
                self.best_miou = jac
                #self.time_tuning_model.save(f"checkpoints/model_best_miou_epoch_{epoch}.pth")
                save_path = os.path.join(checkpoint_dir, f"model_best_miou_epoch_{epoch}_{self.time_tuning_model.model_type}_{self.time_tuning_model.training_set}_FIXED_reg2.pth")
                self.time_tuning_model.save(save_path)
                print(f"Model saved with mIoU: {self.best_miou} at epoch {epoch}")
            # elif jac > 0.165:
            #     save_path = os.path.join(checkpoint_dir, f"model_best_miou_epoch_{epoch}_{self.time_tuning_model.model_type}_{self.time_tuning_model.training_set}_justloaded.pth")
            #     self.time_tuning_model.save(save_path)
            #     print(f"Model saved with mIoU: {self.best_miou} at epoch {epoch} -- not the best")
            # save latest model checkpoint nonetheless
            # should always overwrite
            save_path_latest = os.path.join(checkpoint_dir, f"latest_model_{self.time_tuning_model.model_type}_{self.time_tuning_model.training_set}_FIXED_reg2.pth")
            self.time_tuning_model.save(save_path_latest)
    

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
    import torch
    print(torch.version.cuda)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('starting run')
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
    logger = wandb.init(project=project_name, group=d1, mode="online", job_type='debug_clustering_ytvos', config=config)
    rand_color_jitter = video_transformations.RandomApply([video_transformations.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.8)
    data_transform_list = [rand_color_jitter, video_transformations.RandomGrayscale(), video_transformations.RandomGaussianBlur()]
    data_transform = video_transformations.Compose(data_transform_list)
    video_transform_list = [video_transformations.Resize(224), video_transformations.RandomResizedCrop((224, 224)), video_transformations.RandomHorizontalFlip(), video_transformations.ClipToTensor(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])] #video_transformations.RandomResizedCrop((224, 224))
    video_transform = video_transformations.Compose(video_transform_list)
    num_clips = 1
    num_clip_frames = 4
    if args.training_set == 'co3d':
        num_clip_frames = 4 # co3d has frames split over 5 so might make more sense
    regular_step = 1
    print('setup trans done')
    transformations_dict = {"data_transforms": data_transform, "target_transforms": None, "shared_transforms": video_transform}
    prefix = args.prefix_path
    if args.training_set == 'ytvos':
        data_path = os.path.join(prefix, "train/JPEGImages")
        annotation_path = os.path.join(prefix, "train/Annotations")
        meta_file_path = os.path.join(prefix, "train/meta.json")
    elif args.training_set == 'co3d':
        data_path = os.path.join(prefix, "zips")
        annotation_path = os.path.join(prefix, "zips")
        meta_file_path = os.path.join(prefix, "zips")
    print(data_path)
    print(annotation_path)
    print(meta_file_path)
    path_dict = {"class_directory": data_path, "annotation_directory": annotation_path, "meta_file_path": meta_file_path}
    sampling_mode = SamplingMode.DENSE
    if args.training_set == 'ytvos':
        video_data_module = VideoDataModule("ytvos", path_dict, num_clips, num_clip_frames, sampling_mode, regular_step, batch_size, num_workers)
    elif args.training_set == 'co3d':
        video_data_module = VideoDataModule("co3d", path_dict, num_clips, num_clip_frames, sampling_mode, regular_step, batch_size, num_workers)
    video_data_module.setup(transformations_dict)
    video_data_module.make_data_loader()
    spair_dataset = None
    if args.spair_val:
        print(f'spair_data_path: {args.spair_path}')
        vp_diff = None
        spair_dataset = SPairDataset(
            root=args.spair_path,
            split="test",
            use_bbox=False,
            image_size=224,
            image_mean="imagenet",
            class_name= None, # loop over classes in val if we want a per class recall
            num_instances=5000,
            vp_diff=vp_diff,
        )
        print(f'Length of SPair Dataset: {len(spair_dataset)}')
    if args.model_type == 'dino':
        vit_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    elif args.model_type == 'dinov2':
        #vit_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        vit_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg_lc')
        vit_model = vit_model.backbone
        # print(f'Vit model loaded: {vit_model}')
        # print(f'DIR Vit model loaded: {dir(vit_model)}') 
        #vit_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    elif args.model_type == 'registers':
        vit_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg_lc')
        vit_model = vit_model.backbone
        print(hasattr(vit_model, 'forward_features'))
    patch_prediction_model = TimeTuningV2(224, vit_model, logger=logger, model_type=args.model_type, training_set = args.training_set, use_neco_loss=args.use_neco_loss)
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
    dataset = PascalVOCDataModule(batch_size=batch_size, train_transform=val_transforms, val_transform=val_transforms, test_transform=val_transforms, dir = args.pascal_path, num_workers=num_workers)
    dataset.setup()
    test_dataloader = dataset.get_test_dataloader()
    patch_prediction_trainer = TimeTuningV2Trainer(video_data_module, test_dataloader, patch_prediction_model, num_epochs, device, logger, spair_dataset=spair_dataset, spair_val=args.spair_val, use_neco_loss=args.use_neco_loss) 
    patch_prediction_trainer.setup_optimizer(optimization_config)
    patch_prediction_trainer.train()

    # patch_prediction_trainer.visualize()


            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:3")
    #parser.add_argument('--prefix_path', type=str, default="/scratch-shared/isimion1/timet")
    parser.add_argument('--prefix_path', type=str, default="/projects/2/managed_datasets/co3d/")
    parser.add_argument('--pascal_path', type=str, default="/scratch-shared/isimion1/pascal/VOCSegmentation")
    parser.add_argument('--spair_path', type=str, default="/home/isimion1/probe3d/data/SPair-71k")
    parser.add_argument('--ucf101_path', type=str, default="/scratch-shared/isimion1/timet/train")
    parser.add_argument('--clip_durations', type=int, default=2)
    parser.add_argument('--training_set', type=str, choices=['ytvos', 'co3d'], default='ytvos')
    parser.add_argument('--spair_val', type=float, default=False)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=12)#12 put 0 for debugging
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--num_epochs', type=int, default=800)
    parser.add_argument('--crop_size', type=int, default=64)
    parser.add_argument('--crop_scale_tupple', type=tuple, default=(0.3, 1))
    parser.add_argument('--model_type', type=str, choices=['dino', 'dinov2', 'registers'], default='dinov2', help='Select model type: dino or dinov2')
    parser.add_argument('--masking_ratio', type=float, default=1)
    parser.add_argument('--same_frame_query_ref', type=bool, default=False)
    parser.add_argument("--explaination", type=str, default="clustering, every other thing is the same; except the crop and reference are not of the same frame. and num_crops =4")
    parser.add_argument("--use_neco_loss", type=bool, default=False)
    args = parser.parse_args()
    run(args)