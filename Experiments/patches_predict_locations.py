import torch

from data_loader import PascalVOCDataModule



class PredictionHead(torch.nn.Module):
    def __init__(self, input_dim=384, hidden_layers=[], output_dim=196):
        super(PredictionHead, self).__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.layers = []
        self.layers.append(torch.nn.Linear(self.input_dim, self.hidden_layers[0]))
        for i in range(len(self.hidden_layers) - 1):
            self.layers.append(torch.nn.Linear(self.hidden_layers[i], self.hidden_layers[i+1]))
            self.layers.append(torch.nn.GLU())
        self.layers.append(torch.nn.Linear(self.hidden_layers[-1], self.output_dim))
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
        spatial_features = self.dino.get_intermediate_layers(x)
        return self.prediction_head(spatial_features)
    
    def set_up_optimizer(self, optimizer, lr, scheduler=None):
        self.optimizer = optimizer(self.parameters(), lr=lr)
        if scheduler is not None:
            self.scheduler = scheduler(self.optimizer)
    
    def train_step(self, x, y):
        self.optimizer.zero_grad()
        y_hat = self.forward(x)
        y = torch.arange(0, 196).repeat(x.shape[0]).to(y.device)
        loss = self.criterion(y_hat, y)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item()

    def train(self):
        self.dino.eval()
        self.prediction_head.train()
    
    def eval(self):
        self.dino.eval()
        self.prediction_head.eval()


class Trainer():
    def __init__(self, model, data_module, device):
        self.model = model
        self.data_module = data_module
        self.device = device
        self.model.to(self.device)
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
    
    def train(self, epochs):
        for epoch in range(epochs):
            self.train_on_epoch(epoch)
            if epoch % 5 == 0:
                self.evaluate()
    
    def evaluate(self):
        eval_loss = 0
        self.model.eval()
        for i, batch in enumerate(self.val_loader):
            x, y = batch
            y = torch.arange(0, 196).repeat(x.shape[0]).to(y.device)
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat = self.model.forward(x)
            loss = self.criterion(y_hat, y)
            eval_loss += loss.item()
        eval_loss /= len(self.val_loader)
        print(f"Evaluation loss : {eval_loss}")

    



if __name__ == "__main__":
    dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    dino.eval()
    prediction_head = PredictionHead(input_dim=384, hidden_layers=[384, 384], output_dim=196)
    model = DINOPatchPredictor(dino, prediction_head)
    model.set_up_optimizer(torch.optim.Adam, lr=1e-4)
    data_module = PascalVOCDataModule(batch_size=4, train_transform=None, val_transform=None, test_transform=None)
    trainer = Trainer(model, data_module, "cuda")
    trainer.train(epochs=100)
