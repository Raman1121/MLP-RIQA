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

import yaml
from pprint import pprint



# ==================================================================== #

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
test_df = pd.read_csv(yaml_data['test']['test_df_path'])

main_df['image'] = main_df['image'].apply(lambda x: str(yaml_data['dataset']['root_path']+'final_train/train/'+x))
test_df['image'] = test_df['image'].apply(lambda x: str(yaml_data['dataset']['root_path']+'final_test/test/'+x))

# Creating training and validation splits
train_df, val_df = train_test_split(main_df, test_size=yaml_data['train']['validation_split'],
                                    random_state=yaml_data['train']['seed'])

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print("Length of Main dataset: ", len(main_df))

#Creating Datasets
train_dataset = retinopathy_dataset.RetinopathyDataset(df=train_df, categorical_partitition=True,
                                                cat_labels_to_include=yaml_data['train']['cat_labels'], transforms=train_transform)

val_dataset = retinopathy_dataset.RetinopathyDataset(df=val_df, categorical_partitition=True,
                                                cat_labels_to_include=yaml_data['validation']['cat_labels'], transforms=train_transform)

test_dataset = retinopathy_dataset.RetinopathyDataset(df=test_df, categorical_partitition=True,
                                                cat_labels_to_include=yaml_data['test']['cat_labels'], transforms=train_transform)



print("Length of Training dataset: ", train_dataset.__len__())
print("Length of Validation dataset: ", val_dataset.__len__())
print("Length of Test dataset: ", test_dataset.__len__())

#Creating Dataloaders
# train_loader = DataLoader(train_dataset, batch_size=yaml_data['train']['batch_size'], shuffle=True, num_workers=12)
val_loader = DataLoader(val_dataset, batch_size=yaml_data['train']['batch_size'], shuffle=False, num_workers=12)
test_loader = DataLoader(test_dataset, batch_size=yaml_data['train']['batch_size'], shuffle=False, num_workers=12)

#dm = retinopathy_dataset.LightningRetinopathyDataset(train_dataset, val_dataset, test_dataset, yaml_data['train']['batch_size'])
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

