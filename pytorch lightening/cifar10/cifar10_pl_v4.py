# regular imports
import os
import re
import numpy as np
import argparse
import logging
import time
import json
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-lightning"])

# pytorch related imports
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.optim.lr_scheduler import OneCycleLR

# lightning related imports
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
#from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir: str = './data'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262))
        ])
        
        self.dims = (3, 32, 32)
        self.num_classes = 10

    def prepare_data(self):
        # download 
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=4)
        
def create_model():
    model = torchvision.models.resnet34(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model
    
class LitResnet34(pl.LightningModule):
    def __init__(self, batch_size, learning_rate=2e-4):
        super().__init__()
        
        # log hyperparameters
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.model = create_model()
  
    # will be used during inference
    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        
        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True) #prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, logger=True) #prog_bar=True)
        
        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    # logic for a single testing step
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        
        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9,weight_decay=5e-4,)
        steps_per_epoch = 45000 // self.batch_size
        scheduler_dict = { "scheduler": OneCycleLR(optimizer, 0.1, epochs=self.trainer.max_epochs, steps_per_epoch=steps_per_epoch,),
                           "interval": "step",}
        #return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
        return optimizer

def _train(args):
    
    # Init our data pipeline
    dm = CIFAR10DataModule(batch_size=32)
    
    # To access the x_dataloader we need to call prepare_data and setup.
    dm.prepare_data()
    dm.setup()
    
    # Init our model
    print('Batch-size to be trained:', dm.batch_size)
    model = LitResnet34(dm.batch_size)
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=20,
        verbose=False,
        mode='min'
        )    

    # check if checkpoint exists
    if not os.path.isfile(args.checkpoint_path + '/checkpoint.pth'):
        epoch_number=0
    else:
        #model, optimizer, epoch_number = _load_checkpoint(model, optimizer, args)
        model = model.load_from_checkpoint(args.checkpoint_path + '/checkpoint.pth')
        print(f'Model loaded from {args.checkpoint_path}')
   
    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=2,
                        progress_bar_refresh_rate=20, 
                        gpus=0, 
                        logger=TensorBoardLogger("lightning_logs/", name="resnet34_jan18_v0"),
                        enable_model_summary=True,
                        callbacks=[early_stop_callback,
                                ModelSummary(max_depth=-1)])     
       
    # Train the model again
    trainer.fit(model, dm)

    # Evaluate the model on the held out test set
    trainer.test(model, dm)   
    
    #_save_checkpoint(model, optimizer, epoch, loss, args)
    trainer.save_checkpoint(args.checkpoint_path + '/checkpoint.pth')
    print(f'Model saved to {args.checkpoint_path}')

    return _save_model(model, args.model_dir)     

def _save_model(model, model_dir):
    print("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html    
    torch.save(model.cpu().state_dict(), path)
    print(f'Model saved to :{path}')

def _save_checkpoint(model, optimizer, epoch, loss, args):
    print(f'epoch : {epoch+1}, loss : {loss}')
    checkpointing_path = args.checkpoint_path + '/checkpoint.pth'
    print(f'saving the checkpoint : {checkpointing_path}')
    torch.save({
        'epoch':epoch+1,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'loss':loss,
    }, checkpointing_path)
    
# function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def _load_checkpoint(model, optimizer, args):
    print('-------')
    print('checkpoint file found !')
    print(f"loading checkpoint from {args.checkpoint_path}'/checkpoint.pth'")
    checkpoint = torch.load(args.checkpoint_path + '/checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_number = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f'checkpoint file loaded, epoch_number : {epoch_number} & loss : {loss}')
    print(f'Resuming training from epoch : {epoch_number + 1}')
    print('-------')
    return model, optimizer, epoch_number

def model_fn(model_dir):
    print("model_fn")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    model = LitResnet34(batch_size)
    if torch.cuda.device_count() > 1:
        print("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
        
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
        print(f'Model Loaded successfully from {model_dir}')
    return model.to(device)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        metavar="W",
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="E",
        help="number of total epochs to run (default: 2)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, metavar="BS", help="batch size (default: 4)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, metavar="M", help="momentum (default: 0.9)"
    )
    parser.add_argument(
        "--dist_backend", type=str, default="gloo", help="distributed backend (default: gloo)"
    )

    parser.add_argument("--hosts", type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument("--current-host", type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument("--model-dir", type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument("--data-dir", type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument("--checkpoint-path", type=str, default="/opt/ml/checkpoints")
    parser.add_argument("--num-gpus", type=int, default=os.environ['SM_NUM_GPUS'])
    
    args, _ = parser.parse_known_args()
    print('Printing Args :', args)

    _train(parser.parse_args())      