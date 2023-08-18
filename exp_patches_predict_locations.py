import torch
import sys
from data_loader import PascalVOCDataModule
from image_transformations import Compose, RandomResizedCrop, RandomHorizontalFlip, Resize
from torchvision import transforms
import torchvision.transforms as trn
import wandb

project_name = "TimeTuning_v2"

class PredictionHead(torch.nn.Module):
    def __init__(self, input_dim=384, hidden_layers=[], output_dim=196):
        super(PredictionHead, self).__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.layers = []
        if len(self.hidden_layers) != 0:
            self.layers.append(torch.nn.Linear(self.input_dim, self.hidden_layers[0]))
            self.layers.append(torch.nn.GELU())
            for i in range(len(self.hidden_layers) - 1):
                self.layers.append(torch.nn.Linear(self.hidden_layers[i], self.hidden_layers[i+1]))
                self.layers.append(torch.nn.GELU())
            self.layers.append(torch.nn.Linear(self.hidden_layers[-1], self.output_dim))
        else:
            self.layers.append(torch.nn.Linear(self.input_dim, self.output_dim))
        self.layers = torch.nn.Sequential(*self.layers)

        self.optimizer = None
        self.scheduler = None
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.layers(x)



class DINOPatchPredictor(torch.nn.Module):
    def __init__(self, dino, prediction_head):
        super(DINOPatchPredictor, self).__init__()
        self.dino = dino
        self.prediction_head = prediction_head
        self.optimizer = None
        self.scheduler = None
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def forward(self, x):
        spatial_features = self.dino.get_intermediate_layers(x)[0][:, 1:, :]
        bs, np, dim = spatial_features.shape
        spatial_features = spatial_features.flatten(0, 1)
        return self.prediction_head(spatial_features).reshape(bs, np, -1)
    
    def set_up_optimizer(self, optimizer, lr, scheduler=None):
        self.optimizer = optimizer(self.parameters(), lr=lr)
        if scheduler is not None:
            self.scheduler = scheduler(self.optimizer)
    
    def train_step(self, x, y):
        self.optimizer.zero_grad()
        y_hat = self.forward(x)
        y = torch.arange(0, 196).unsqueeze_(0).repeat(x.shape[0], 1).to(y.device)
        y = y.flatten(0, 1)
        y_hat = y_hat.flatten(0, 1)
        loss = self.criterion(y_hat, y)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item()

    def val_step(self, x, y):
        y_hat = self.forward(x)
        y = torch.arange(0, 196).unsqueeze_(0).repeat(x.shape[0], 1).to(y.device)
        y = y.flatten(0, 1)
        y_hat = y_hat.flatten(0, 1)
        loss = self.criterion(y_hat, y)
        return loss.item()

    def train(self):
        self.dino.eval()
        self.prediction_head.train()
    
    def eval(self):
        self.dino.eval()
        self.prediction_head.eval()


class Trainer():
    def __init__(self, model, data_module, device, logger=None):
        self.model = model
        self.data_module = data_module
        self.device = device
        self.model.to(self.device)
        self.logger = logger
        self.train_loader = self.data_module.get_train_dataloader()
        self.val_loader = self.data_module.get_val_dataloader()
        self.test_loader = self.data_module.get_test_dataloader()
        self.model.train()
    

    def train_on_epoch(self, epoch):
        for i, batch in enumerate(self.train_loader):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            loss = self.model.train_step(x, y)
            print(f"Epoch {epoch} - Batch {i} - Loss : {loss}")
            if self.logger is not None:
                self.logger.log({"train_loss": loss})
    
    def train(self, epochs):
        for epoch in range(epochs):
            self.train_on_epoch(epoch)
            if epoch % 1 == 0:
                self.evaluate()
    
    def evaluate(self):
        eval_loss = 0
        self.model.eval()
        for i, batch in enumerate(self.val_loader):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            loss = self.model.val_step(x, y)
            eval_loss += loss
        eval_loss /= len(self.val_loader)
        print(f"Evaluation loss : {eval_loss}")
        if self.logger is not None:
            self.logger.log({"eval_loss": eval_loss})

    



if __name__ == "__main__":
    logger = wandb.init(project=project_name, group="data_loader", tags="PascalVOCDataModule", job_type="eval")
    dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    dino.eval()
    prediction_head = PredictionHead(input_dim=384, hidden_layers=[], output_dim=196)
    model = DINOPatchPredictor(dino, prediction_head)
    model.set_up_optimizer(torch.optim.Adam, lr=1e-4)
    input_size = 224

    min_scale_factor = 0.5
    max_scale_factor = 2.0
    brightness_jitter_range = 0.1
    contrast_jitter_range = 0.1
    saturation_jitter_range = 0.1
    hue_jitter_range = 0.1

    brightness_jitter_probability = 0.5
    contrast_jitter_probability = 0.5
    saturation_jitter_probability = 0.5
    hue_jitter_probability = 0.5

    # Create the transformation
    image_train_transform = trn.Compose([
        trn.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=0.5),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])
    ])

    shared_train_transform = Compose([
        RandomResizedCrop(size=(input_size, input_size), scale=(min_scale_factor, max_scale_factor), ratio=(1, 1)),
        # RandomHorizontalFlip(probability=0.1),
    ])

    image_val_transform = trn.Compose([trn.Resize((input_size, input_size)), trn.ToTensor(), trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])])
    shared_val_transform = Compose([
        Resize(size=(input_size, input_size)),
    ])
    # target_train_transform = trn.Compose([trn.Resize((224, 224), interpolation=trn.InterpolationMode.NEAREST), trn.ToTensor()])
    train_transforms = {"img": image_train_transform, "target": None, "shared": shared_train_transform}
    val_transforms = {"img": image_val_transform, "target": None , "shared": shared_val_transform}

    data_module = PascalVOCDataModule(batch_size=128, train_transform=train_transforms, val_transform=val_transforms, test_transform=val_transforms)
    data_module.setup()
    trainer = Trainer(model, data_module, "cuda:0")
    trainer.train(epochs=100)
    logger.finish()
