import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import Adam

from dataset import retinopathy_dataset

from pytorch_lightning.core.lightning import LightningModule
import torchmetrics
from torch.nn.functional import cross_entropy

class RetinopathyClassificationModel(LightningModule):
    def __init__(self, encoder='resnet50', pretrained=True, num_classes=5, lr=1e-3):
        super().__init__()

        self.encoder = encoder
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()
        

        if(self.encoder == 'resnet50'):
            self.model = models.resnet50(pretrained=self.pretrained)

        elif(self.encoder == 'resnet18'):
            self.model = models.resnet18(pretrained=self.pretrained)
        
        #Freezing the model layers
        for param in self.model.parameters():
            param.requires_grad = False

        #Attaching a new classification layer on the top of the model
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.num_classes)
        
    def training_step(self, batch, batch_idx):
        
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)

        #Training metrics
        preds = torch.argmax(logits, dim=1)
        acc = torchmetrics.functional.accuracy(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)

        #Validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = torchmetrics.functional.accuracy(preds, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)

        #Validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = torchmetrics.functional.accuracy(preds, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        
        optimizer = Adam(self.model.fc.parameters(), lr=self.lr)
        return optimizer