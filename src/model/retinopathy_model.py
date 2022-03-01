import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR, ReduceLROnPlateau

from dataset import retinopathy_dataset

from pytorch_lightning.core.lightning import LightningModule
import torchmetrics
from torch.nn.functional import cross_entropy
import warnings

class RetinopathyClassificationModel(LightningModule):
    def __init__(self, encoder='resnet50', pretrained=True, num_classes=5, learning_rate=1e-3, lr_scheduler='cyclic', train_all_layers=False, do_finetune=False):
        super().__init__()

        self.encoder = encoder
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.loss = nn.CrossEntropyLoss()
        self.lr_scheduler = lr_scheduler

        if(do_finetune):
            train_all_layers = True
            self.pretrained = True

        if(self.encoder == 'resnet50'):
            self.backbone = models.resnet50(pretrained=self.pretrained)

        elif(self.encoder == 'resnet18'):
            self.backbone = models.resnet18(pretrained=self.pretrained)
        
        if(not train_all_layers):

            #Freezing the model layers
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        if(train_all_layers):
            if(pretrained):
                print("Model is in fine-tuning mode. [Using ImageNet weights and training all layers]")

        #Attaching a new classification layer on the top of the model
        self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, self.num_classes)
        self.model = self.backbone
        
    def training_step(self, batch, batch_idx):
        
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)

        #Training metrics
        preds = torch.argmax(logits, dim=1)
        acc = torchmetrics.functional.accuracy(preds, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        #print("Training Loss: {} Training Accuracy: {}".format(loss, acc))

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)

        #Validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = torchmetrics.functional.accuracy(preds, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        #print("Validation Loss: {} Validation Accuracy: {}".format(loss, acc))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)

        #Validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = torchmetrics.functional.accuracy(preds, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        
        #optimizer = Adam(self.model.fc.parameters(), lr=self.learning_rate)
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        if(self.lr_scheduler == 'none'):
            return optimizer
        elif(self.lr_scheduler == 'cyclic'):
            scheduler = CyclicLR(optimizer, base_lr=self.learning_rate, max_lr=0.1, cycle_momentum=False, verbose=True)
        elif(self.lr_scheduler == 'cosine'):
            scheduler = CosineAnnealingLR(optimizer, T_max=1000, verbose=True)
        elif(self.lr_scheduler == 'reduce_plateau'):
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, min_lr=1e-6, patience=5, verbose=True)

        return {"optimizer": optimizer, 
                "lr_scheduler":{
                    "scheduler": scheduler,
                    "monitor": 'val_loss'
                }}
