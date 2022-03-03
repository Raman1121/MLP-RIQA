from turtle import title
import torch
import numpy as np
import os
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR, ReduceLROnPlateau

from dataset import retinopathy_dataset

from pytorch_lightning.core.lightning import LightningModule
import torchmetrics
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from torch.nn.functional import cross_entropy

from torchsummary import summary
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import warnings
import itertools


class RetinopathyClassificationModel(LightningModule):
    def __init__(self, encoder='resnet50', pretrained=True, num_classes=5, learning_rate=1e-3, 
                 lr_scheduler='cyclic', train_all_layers=False, do_finetune=False, plot_save_dir=None):
        super().__init__()

        self.encoder = encoder
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.loss = nn.CrossEntropyLoss()
        self.lr_scheduler = lr_scheduler
        self.ground_truths = []
        self.predictions = []
        self.classes = ['Severity 0', 'Severity 1', 'Severity 2', 'Severity 3', 'Severity 4']
        self.plot_save_dir = plot_save_dir
        self.experiment = str(self.plot_save_dir.split('/')[-1])

        self.fc1_features = 512

        if(do_finetune):
            #Finetuning requires updating the whole model with pretrained weights.
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

        

        self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, self.num_classes)
        self.model = self.backbone

        #Creating my module to add on top of the backbone.
        #Attaching 2 new classification layers on the top of the model

        # self.mymodule = nn.Sequential(
        #                     nn.Linear(self.backbone.fc.in_features, self.fc1_features),
        #                     nn.Linear(self.fc1_features, self.num_classes)
        # )

        # self.model = nn.Sequential(self.backbone, self.mymodule)
        
    def training_step(self, batch, batch_idx):
        
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)

        #Training metrics
        preds = torch.argmax(logits, dim=1)
        acc = torchmetrics.functional.accuracy(preds, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        #print("Training Loss: {} Training Accuracy: {}".format(loss, acc))

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)

        #Validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = torchmetrics.functional.accuracy(preds, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        #print("Validation Loss: {} Validation Accuracy: {}".format(loss, acc))

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

        y_list = y.tolist()
        preds_list = preds.tolist()

        self.ground_truths.extend(y_list)
        self.predictions.extend(preds_list)

        # for i in range(len(y)):
        #     self.ground_truths.append(y_list[i])
        #     self.predictions.append(preds_list[i])
        
        return loss

    def plot_confusion_matrix(self, ground_truths, predictions, classes, 
                              title='Confusion matrix', cmap=plt.cm.Blues, normalize=False, plot_save_dir=None):

        print(" ######################## PLOTTING CM NOW ########################")
        cm = confusion_matrix(ground_truths, predictions)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        print(cm)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        try:
            plt.savefig(os.path.join(plot_save_dir, 'Confusion_Matrix.png'))
        except:
            os.makedirs(plot_save_dir)
            plt.savefig(os.path.join(plot_save_dir, 'Confusion_Matrix.png'))

    def test_epoch_end(self, outputs):
        
        #Plot confusion matrix when testing ends
        cm_title = 'Confusion Matrix for ' + self.experiment
        self.plot_confusion_matrix(ground_truths = self.ground_truths, predictions = self.predictions, 
                              classes = self.classes, plot_save_dir=self.plot_save_dir, title=cm_title)

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
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, min_lr=1e-6, verbose=True)

        return {"optimizer": optimizer, 
                "lr_scheduler":{
                    "scheduler": scheduler,
                    "monitor": 'val_loss'
                }}

    
