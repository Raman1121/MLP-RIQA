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

# ============================================================ #

with open('config_train_DR.yaml') as file:
    yaml_data = yaml.safe_load(file)

if(yaml_data['train']['verbose']):
    pprint("CONFIG HYPERPARAMS: ")
    pprint(yaml_data)
    print('\n')

if(yaml_data['wandb']['run_name'] != ''):
    #If name is provided, initialize the logger with a name
    wandb_logger = WandbLogger(name = yaml_data['wandb']['run_name'], log_model=True)
else:
    #Else, initialize without a name
    wandb_logger = WandbLogger(log_model=True)
    
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
                                                            num_classes=yaml_data['train']['num_classes'], 
                                                            learning_rate=yaml_data['train']['lr'])

trainer = pl.Trainer(gpus=yaml_data['train']['gpus'], max_epochs=yaml_data['train']['epochs'], 
                     logger=wandb_logger, 
                     default_root_dir=os.path.join(yaml_data['save_model']['directory'], yaml_data['save_model']['experiment']))  

trainer.fit(classifier, dm)

#Testing on the validation set
trainer.test(classifier, dataloaders=val_loader)
