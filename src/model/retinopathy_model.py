import torch
from torch import nn
import torchvision.models as models
from torch.optim import Adam

from dataset import retinopathy_dataset

from pytorch_lightning.core.lightning import LightningModule
from torchmetrics import Accuracy
from torch.nn.functional import cross_entropy

class RetinopathyClassificationModel(LightningModule):
    def __init__(self, encoder='resnet50', pretrained=True, num_classes=5, lr=1e-3):
        super().__init__()

        self.encoder = encoder
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.lr = lr

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
        # return the loss given a batch: this has a computational graph attached to it: optimization
        x, y = batch
        preds = self.model(x)
        loss = cross_entropy(preds, y)
        self.log('train_loss', loss)  # lightning detaches your loss graph and uses its value
        #self.log('train_acc', Accuracy(preds, y))
        return loss

    def configure_optimizers(self):
        # return optimizer
        optimizer = Adam(self.model.fc.parameters(), lr=self.lr)
        return optimizer