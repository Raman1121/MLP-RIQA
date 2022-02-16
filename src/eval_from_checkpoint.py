import torch
import os
from dataset import retinopathy_dataset
from model import retinopathy_model

from model import retinopathy_model
from torch.utils.data import DataLoader
from torchvision import transforms as T

import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import train_test_split
import yaml
from pprint import pprint
import pytorch_lightning as pl

import logging
logging.getLogger("lightning").setLevel(logging.ERROR)

# =============================================================================================== # 

with open('config_val.yaml') as file:
    yaml_data = yaml.safe_load(file)

pprint(yaml_data)

#################### DEFINE CONSTANTS ####################

DATASET = yaml_data['run_evaluation_on']
CHECKPOINT_PATH = yaml_data['checkpoint_path']
NUM_CLASSES = yaml_data['num_classes']
ENCODER = yaml_data['encoder']
TRAIN_CAT_LABELS = yaml_data['training']['cat_labels']
VAL_CAT_LABELS = yaml_data['validation']['cat_labels']
TEST_CAT_LABELS = yaml_data['testing']['cat_labels']
DATASET_ROOT_PATH = yaml_data['dataset']['root_path']
TRAIN_DF_PATH = yaml_data['dataset']['train_df_path']
TEST_DF_PATH = yaml_data['dataset']['test_df_path']
SEED = yaml_data['seed']
BATCH_SIZE = yaml_data['batch_size']
GPUS = yaml_data['gpus']
VALIDATION_SPLIT = 0.3

##########################################################

transforms = T.Compose([
    T.ToPILImage(),
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

main_df = pd.read_csv(TRAIN_DF_PATH)
test_df = pd.read_csv(TEST_DF_PATH)

main_df['image'] = main_df['image'].apply(lambda x: str(DATASET_ROOT_PATH+'final_train/train/'+x))
test_df['image'] = test_df['image'].apply(lambda x: str(DATASET_ROOT_PATH+'final_test/test/'+x))

# Creating training and validation splits
train_df, val_df = train_test_split(main_df, test_size=VALIDATION_SPLIT,
                                    random_state=SEED)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

backbone = retinopathy_model.RetinopathyClassificationModel(encoder=ENCODER, pretrained=True, 
                                                            num_classes=NUM_CLASSES, lr_scheduler='none'
                                                            )

classifier = backbone.load_from_checkpoint(checkpoint_path=CHECKPOINT_PATH)

print('Model loaded from checkpoint successfully!')

#print('eval file')

trainer = pl.Trainer(gpus=GPUS)

if(DATASET == 'validation'):
    print('Performing evaluation on validation dataset')
    val_dataset = retinopathy_dataset.RetinopathyDataset(df=val_df, categorical_partitition=True,
                                                         cat_labels_to_include=VAL_CAT_LABELS, transforms=transforms)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)
    trainer.test(classifier, dataloaders=val_loader)

elif(DATASET == 'train'):
    print('Performing evaluation on training dataset')
    train_dataset = retinopathy_dataset.RetinopathyDataset(df=train_df, categorical_partitition=True,
                                                         cat_labels_to_include=TRAIN_CAT_LABELS, transforms=transforms) 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)
    trainer.test(classifier, dataloaders=train_loader)

elif(DATASET == 'test'):
    print('Performing evaluation on test dataset')
    test_dataset = retinopathy_dataset.RetinopathyDataset(df=test_df, categorical_partitition=True,
                                                         cat_labels_to_include=TEST_CAT_LABELS, transforms=transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)
    trainer.test(classifier, dataloaders = test_loader)
