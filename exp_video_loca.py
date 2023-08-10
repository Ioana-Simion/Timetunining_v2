import argparse
from data_loader import PascalVOCDataModule
import torch
import torchvision.transforms as trn
import wandb
import torch.nn.functional as F
from models import mae_vit_base_patch16
import torch.optim as optim
import datetime
from clustering import PerDatasetClustering, PerClipClustering, PerFrameClustering
from eval_metrics import PredsmIoU
from my_utils import show_trainable_paramters
from optimizer import DINOMAEOptimizer
from torch.optim import AdamW
from evaluator import LinearFinetune
import copy
from my_utils import process_attentions


project_name = "DINO_MAE"





class DinoMAE_Trainer():
    def __init__(self, model, optimizer, data_set, logger, device) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = data_set.get_train_dataloader()
        self.val_dataloader = data_set.get_val_dataloader()
        self.test_dataloader = data_set.get_test_dataloader()
        self.device = device
        self.logger = logger
        self.model = self.model.to(self.device)
    
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            if epoch % 50 == 50:
                self.linear_segmentation_validation()
            self.train_epoch(epoch)
    

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
            for i, (x, y) in enumerate(self.val_dataloader):
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
    
    
    def linear_segmentation_validation(self, train_epoch=25):
        spatial_resolution = self.model.get_dino_feature_spatial_resolution()
        drop_at = 20
        total_iters = len(self.train_dataloader) * drop_at
        cloned_model = copy.deepcopy(self.model)


        ## keep a dictionary of a few parameters of the model and later check if they are changed
        ########################################################
        # dict = {}
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         dict[name] = param.data.clone()

        ########################################################
    
        linear_evaluator = LinearFinetune(cloned_model, train_epoch=25, num_classes=21, lr=0.01, input_size=224, spatial_res=spatial_resolution, val_iters=20,
                 drop_at=total_iters, arch="vit_small", head_type="lc", device=self.device)
        
        ########################################################
        ## check if the parameters are changed
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         assert torch.equal(param.data, dict[name])

        ########################################################
        

        print("==============================================================")
        for j in range(train_epoch):
            for i, (x, y) in enumerate(self.train_dataloader):
                loss = linear_evaluator.train_step((x, y))
                print('linear_eval_loss', loss)
            if j % 10 == 0:
                for i, (x, y) in enumerate(self.val_dataloader):
                    linear_evaluator.validation_step((x, y))
                miou = linear_evaluator.validation_epoch_end()
                print('miou_val', round(miou, 6))
            



    def train_epoch(self, epoch):
        self.model.train()
        for i, (x, y) in enumerate(self.train_dataloader):
            target = (y * 255).long().to(self.device)
            img = x.to(self.device)
            ## create an all 1 img normalized with imagenet stats
            # all_1_img = torch.ones_like(img)
            # all_1_img = trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])(all_1_img)
            # all_1_img = all_1_img.to(self.device)
            cls_loss, spatial_loss = self.model.train_step(img, target)
            loss = cls_loss + 0 * spatial_loss
            self.logger.log({"cls_loss": cls_loss, "spatial_loss": spatial_loss, "loss": loss})
            self.optimizer.zero_grad()
            loss.backward()
            lr = self.optimizer.get_lr()
            weight_decay = self.optimizer.get_weight_decay()
            self.logger.log({"lr": lr, "weight_decay": weight_decay})
            self.optimizer.step()


def run(args):

    ## the date of the run
    my_run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    ## wandb init
    group = args.group
    tags = args.tags
    job_type = args.job_type
    ## hyperparameters
    lr = args.lr
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_epochs = args.num_epochs
    mask_ratio = args.mask_ratio
    device = args.gpu
    init_lr = args.init_lr
    peak_lr = args.peak_lr
    decay_half_life = args.decay_half_life
    warmup_steps = args.warmup_steps
    grad_norm_clipping = args.grad_norm_clipping
    init_weight_decay = args.init_weight_decay
    peak_weight_decay = args.peak_weight_decay
    

    if torch.cuda.is_available():
        print ("cuda is available")
        device = torch.device(f"cuda:{device}")
    else:
        print ("cuda is not available")
        device = torch.device("cpu")

    logger = wandb.init(name=my_run_name ,project=project_name, group=group, tags=tags, job_type=job_type, config=args)

    image_train_transform = trn.Compose([trn.Resize((224, 224)), trn.ToTensor(), trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])])
    target_train_transform = trn.Compose([trn.Resize((224, 224), interpolation=trn.InterpolationMode.NEAREST), trn.ToTensor()])
    train_transforms = {"img": image_train_transform, "target": target_train_transform}
    dataset = PascalVOCDataModule(batch_size=batch_size, train_transform=train_transforms, val_transform=train_transforms, test_transform=train_transforms, num_workers=num_workers)
    dataset.setup()
    dino_mae = DinoMAE(mask_ratio, dino_version="dino_vits16")
    wandb.watch(dino_mae, log="all", log_freq=10)
    max_iter = (dataset.get_train_dataset_size() // batch_size) * num_epochs
    ## initialize adamw optimizer with weight decay=0.5, lr=1.5e-4 beta1=0.9 beta2=0.95 eps=1e-8 learning rate schedule is cosine decay warmup for 40 epochs

    # optimizer = AdamW(dino_mae.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.95), eps=1e-8)

    optimizer =  DINOMAEOptimizer(dino_mae.parameters(), init_lr, peak_lr, decay_half_life, warmup_steps, grad_norm_clipping, init_weight_decay, peak_weight_decay, max_iter)

    optimizer.setup_optimizer()
    optimizer.setup_scheduler()

    show_trainable_paramters(dino_mae)
    trainer = DinoMAE_Trainer(dino_mae, optimizer, dataset, logger, device)
    trainer.train(num_epochs)
    logger.finish()

if __name__ == "__main__":
    ## parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0004, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
    parser.add_argument("--num_epochs", type=int, default=500, help="num_epochs")
    parser.add_argument("--init_lr", type=float, default=0.0001, help="init_lr")
    parser.add_argument("--peak_lr", type=float, default=0.001, help="peak_lr")
    parser.add_argument("--decay_half_life", type=int, default=10000, help="decay_half")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="warmup_steps")
    parser.add_argument("--grad_norm_clipping", type=bool, default=True, help="grad_norm_clipping")
    parser.add_argument("--init_weight_decay", type=float, default=0.04, help="init_weight_decay")
    parser.add_argument("--peak_weight_decay", type=float, default=0.4, help="peak_weight_decay")
    parser.add_argument("--mask_ratio", type=float, default=0.75, help="new optimizer")
    parser.add_argument("--job_type", type=str, default="debug linear classification v1", help="job_type")
    parser.add_argument("--tags", type=str, default="main loop", help="tags")
    parser.add_argument("--group", type=str, default="main", help="group")
    parser.add_argument("--gpu", type=int, default=1, help="gpu")

    args = parser.parse_args()
    run(args)