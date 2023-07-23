import torch
from models import FeatureExtractor
from data_loader import PascalVOCDataModule
import torchvision.transforms as trn

class HummingbirdEvaluation():
    def __init__(self, feature_extractor, dataset_module, augmentation_epoch, memory_size, device):
        self.feature_extractor = feature_extractor
        self.dataset_module = dataset_module
        self.device = device
        self.augmentation_epoch = augmentation_epoch
        self.memory_size = memory_size
        self.feature_extractor.eval()
        self.feature_extractor = feature_extractor.to(self.device)
        self.memory = self.create_memory()

    def create_memory(self):
        train_loader = self.dataset_module.get_train_dataloader()
        eval_spatial_resolution = self.feature_extractor.eval_spatial_resolution
        memory = []
        num_sampled_features = self.memory_size // (self.dataset_module.get_train_dataset_size() * self.augmentation_epoch)
        for j in range(self.augmentation_epoch):
            for i, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                y = (y * 255).long()
                features = self.feature_extractor.get_intermediate_layer_feats(x)
                input_size = x.shape[-1]
                patch_size = input_size // eval_spatial_resolution
                pathified_gts = self.patchify_gt(y, patch_size) ## (bs, spatial_resolution, spatial_resolution, c*patch_size*patch_size)
                for k, gt in enumerate(pathified_gts):
                    class_frequency = self.get_class_frequency(gt)
                    patch_scores = self.get_patch_scores(gt, class_frequency)
                    zero_score_idx = torch.where(patch_scores == 0)
                    none_zero_score_idx = torch.where(patch_scores != 0)
                    patch_scores[zero_score_idx] = 1e-6 
                    ## sample uniform distribution with the size none_zero_score_idx
                    uniform_x = torch.rand(none_zero_score_idx[0])
                    patch_scores[none_zero_score_idx] *= uniform_x
                    feature = features[k]
                    ### select the least num_sampled_features score idndices
                    _, indices = torch.topk(patch_scores.flatten(), num_sampled_features, largest=False)
                    sampled_features = feature[indices]
                    memory.append(sampled_features)

        memory = torch.stack(memory)
        return memory

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



if __name__ == "__main__":
    vit_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    feature_extractor = FeatureExtractor(vit_model)
    image_train_transform = trn.Compose([trn.Resize((224, 224)), trn.ToTensor(), trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])])
    target_train_transform = trn.Compose([trn.Resize((224, 224), interpolation=trn.InterpolationMode.NEAREST), trn.ToTensor()])
    train_transforms = {"img": image_train_transform, "target": target_train_transform}
    dataset = PascalVOCDataModule(batch_size=4, train_transform=train_transforms, val_transform=train_transforms, test_transform=train_transforms)
    dataset.setup()
    evaluator = HummingbirdEvaluation(feature_extractor, dataset, augmentation_epoch=1, memory_size=200000, device="cuda:3")