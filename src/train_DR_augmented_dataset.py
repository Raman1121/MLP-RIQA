import torch
import os
from dataset import retinopathy_dataset
from model import retinopathy_model

from model import retinopathy_model
from torch.utils.data import DataLoader
from torchvision import transforms as T

import pandas as pd
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from sklearn.model_selection import train_test_split
from PIL import Image

import yaml
import random
from pprint import pprint

import logging
logging.getLogger("lightning").setLevel(logging.ERROR)
# ============================================================ #

with open('config_train_DR.yaml') as file:
    yaml_data = yaml.safe_load(file)

if(yaml_data['train']['verbose']):
    pprint("CONFIG HYPERPARAMS: ")
    pprint(yaml_data)
    print('\n')

if(yaml_data['wandb']['run_name'] != ''):
    #If name is provided, initialize the logger with a name
    wandb_logger = WandbLogger(name = yaml_data['wandb']['run_name'], log_model=yaml_data['save_model']['log_model'],
                              save_dir = os.path.join(yaml_data['save_model']['directory'], yaml_data['save_model']['experiment']))
else:
    #Else, initialize without a name
    wandb_logger = WandbLogger(log_model=yaml_data['save_model']['log_model'],
                               save_dir = os.path.join(yaml_data['save_model']['directory'], yaml_data['save_model']['experiment']))

train_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224,224)),
    T.ToTensor(),
])

main_df = pd.read_csv(yaml_data['train']['train_df_path'])
aug_df = pd.read_csv(yaml_data['train']['aug_df_path'])

#Creating validation df
_, val_df = train_test_split(main_df, test_size=yaml_data['train']['validation_split'],
                                    random_state=yaml_data['train']['seed'])

val_df = val_df.reset_index(drop=True)

good_df = main_df.loc[main_df['quality']=='Good'].reset_index(drop=True)
good_df['image'] = good_df['image'].apply(lambda x: str(yaml_data['dataset']['root_path']+'final_train/train/'+x))
val_df['image'] = val_df['image'].apply(lambda x: str(yaml_data['dataset']['root_path']+'final_train/train/'+x))
aug_df['image'] = aug_df['image'].apply(lambda x: str(yaml_data['dataset']['root_path']+'augmented_data/'+x))

#Concat good_df and aug_df
concat_df = pd.concat([good_df, aug_df], axis=0)

#Train dataset consists of both original 'Good' and Augmented 'Good' images
train_dataset = retinopathy_dataset.RetinopathyDataset(df=concat_df, categorical_partitition=True,
                                                cat_labels_to_include=yaml_data['train']['cat_labels'], transforms=train_transform)

#Validaion dataset consists of all kinds of images
val_dataset = retinopathy_dataset.RetinopathyDataset(df=val_df, categorical_partitition=True,
                                                cat_labels_to_include=yaml_data['validation']['cat_labels'], transforms=train_transform)


print("Length of Training dataset: ", train_dataset.__len__())
print("Length of Validation dataset: ", val_dataset.__len__())

#Creating DataLoader
val_loader = DataLoader(val_dataset, batch_size=yaml_data['train']['batch_size'], shuffle=False, num_workers=12)

dm = retinopathy_dataset.LightningRetinopathyDataset(train_dataset, yaml_data['train']['batch_size'])

classifier = retinopathy_model.RetinopathyClassificationModel(encoder=yaml_data['model']['encoder'], pretrained=True, 
                                                            num_classes=yaml_data['train']['num_classes']
                                                            )

if(yaml_data['save_model']['log_model']):

    #Provide a deault_root_dir to save the model in case model logging is True
    trainer = pl.Trainer(gpus=yaml_data['train']['gpus'], 
                        max_epochs=yaml_data['train']['epochs'], 
                        logger=wandb_logger,
                        default_root_dir=os.path.join(yaml_data['save_model']['directory'], 
                        yaml_data['save_model']['experiment']),
                        auto_lr_find=yaml_data['train']['auto_lr_find']
                        )
else:

    #Skip providing a deault_root_dir to save the model in case model logging is False
    trainer = pl.Trainer(gpus=yaml_data['train']['gpus'], 
                        max_epochs=yaml_data['train']['epochs'], 
                        logger=wandb_logger,
                        auto_lr_find=yaml_data['train']['auto_lr_find']
                        )

#Finding the optimal learning rate for model training
if(yaml_data['train']['auto_lr_find']):
    print("~~~~ Finding optimal learning rate before training the model ~~~~~")
    lr_finder = trainer.tuner.lr_find(classifier, dm)
    new_lr = lr_finder.suggestion()
    print("New suggested learning rate is: ", new_lr)
    classifier.hparams.lr = new_lr
else:
    print("~~~~ Using the learning rate provided in the config file ~~~~~")
    classifier.hparams.lr = yaml_data['train']['lr']

trainer.fit(classifier, dm)

#Testing on the validation set
if(yaml_data['validation']['run_validation']):
    trainer.test(classifier, dataloaders=val_loader)