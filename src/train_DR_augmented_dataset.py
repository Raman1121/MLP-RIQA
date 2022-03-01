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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from pl_bolts.callbacks import PrintTableMetricsCallback
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR, ReduceLROnPlateau


from sklearn.model_selection import train_test_split
from PIL import Image

import yaml
import random
import argparse
from pprint import pprint

import logging
logging.getLogger("lightning").setLevel(logging.ERROR)

# ============================================================ #

parser = argparse.ArgumentParser(description='Hyper-parameters management')
parser.add_argument('--experimental_run', type=bool, default=False, help='Experimental run (unit test)')

args = parser.parse_args()

with open('config_train_DR.yaml') as file:
    yaml_data = yaml.safe_load(file)

#################### DEFINE CONSTANTS ####################


#GENERAL CONSTANTS
EXPERIMENTAL_RUN = args.experimental_run
VERBOSE = yaml_data['train']['verbose']

#MODEL CONSTANTS
ENCODER = yaml_data['model']['encoder']
PRETRAINED = yaml_data['model']['pretrained']
TRAIN_ALL_LAYERS = yaml_data['model']['train_all_layers']
DO_FINETUNE = yaml_data['model']['do_finetune']

#TRAINING CONSTANTS
BATCH_SIZE = yaml_data['train']['batch_size']
EPOCHS = yaml_data['train']['epochs']
NUM_CLASSES = yaml_data['train']['num_classes']
TRAIN_DF_PATH = yaml_data['train']['train_df_path']
AUG_DF_PATH = yaml_data['train']['aug_df_path']
GPUS = yaml_data['train']['gpus']
TRAIN_CAT_LABELS = yaml_data['train']['cat_labels']
LR = yaml_data['train']['lr']
VALIDATION_SPLIT = yaml_data['train']['validation_split']
SEED = yaml_data['train']['seed']
AUTO_LR_FIND = yaml_data['train']['auto_lr_find']
LR_SCHEDULING = yaml_data['train']['lr_scheduling']

#VALIDATION CONSTANTS
#RUN_VALIDATION = yaml_data['validation']['run_validation']
VAL_CAT_LABELS = yaml_data['validation']['cat_labels']

#TEST CONSTANTS
TEST_DF_PATH = yaml_data['test']['test_df_path']
TEST_CAT_LABELS = yaml_data['test']['cat_labels']
RUN_EVAL_ON = yaml_data['test']['run_eval_on']

#DATASET CONSTANTS
DATASET_ROOT_PATH = yaml_data['dataset']['root_path']
RIQA_TRAIN_LABELS = yaml_data['dataset']['RIQA_train_labels']
RIQA_TEST_LABELS = yaml_data['dataset']['RIQA_test_labels']

#MODEL SAVING AND LOGGING CONSTANTS
LOG_MODEL = yaml_data['save_model']['log_model']
SAVE_DIR = yaml_data['save_model']['directory']
EXPERIMENT_NAME = yaml_data['save_model']['experiment']

#WANDB CONSTANTS
RUN_NAME = yaml_data['wandb']['run_name']


if(VERBOSE):
    pprint("CONFIG HYPERPARAMS: ")
    pprint(yaml_data)
    print('\n')
'''
if(RUN_NAME != ''):
    #If name is provided, initialize the logger with a name
    wandb_logger = WandbLogger(name = RUN_NAME, log_model=LOG_MODEL,
                              save_dir = os.path.join(SAVE_DIR, EXPERIMENT_NAME))
else:
    #Else, initialize without a name
    wandb_logger = WandbLogger(log_model=LOG_MODEL,
                               save_dir = os.path.join(SAVE_DIR, EXPERIMENT_NAME))
'''

wandb_logger = False

if(EXPERIMENTAL_RUN):
    '''
    Run a unit test on the script
    '''
    
    ENCODER = 'resnet18'
    EPOCHS = 1
    LOG_MODEL = False
    RUN_NAME = ''
    EXPERIMENT_NAME = 'Experimental_Run'
    AUTO_LR_FIND = False
    RUN_VALIDATION = False
    wandb_logger = False

train_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

main_df = pd.read_csv(TRAIN_DF_PATH)
aug_df = pd.read_csv(AUG_DF_PATH)
test_df = pd.read_csv(TEST_DF_PATH)

#Creating validation df
_, val_df = train_test_split(main_df, test_size=VALIDATION_SPLIT,
                                    random_state=SEED)

val_df = val_df.reset_index(drop=True)

good_df = main_df.loc[main_df['quality']=='Good'].reset_index(drop=True)
good_df['image'] = good_df['image'].apply(lambda x: str(DATASET_ROOT_PATH+'final_train/train/'+x))
val_df['image'] = val_df['image'].apply(lambda x: str(DATASET_ROOT_PATH+'final_train/train/'+x))
aug_df['image'] = aug_df['image'].apply(lambda x: str(DATASET_ROOT_PATH+'augmented_data/'+x))
test_df['image'] = test_df['image'].apply(lambda x: str(DATASET_ROOT_PATH+'final_test/test/'+x))

#Concat good_df and aug_df
concat_df = pd.concat([good_df, aug_df], axis=0)

#Train dataset consists of both original 'Good' and Augmented 'Good' images
train_dataset = retinopathy_dataset.RetinopathyDataset(df=concat_df, categorical_partitition=True,
                                                cat_labels_to_include=TRAIN_CAT_LABELS, transforms=train_transform)

#Validaion dataset consists of all kinds of images
val_dataset = retinopathy_dataset.RetinopathyDataset(df=val_df, categorical_partitition=True,
                                                cat_labels_to_include=VAL_CAT_LABELS, transforms=train_transform)

test_dataset = retinopathy_dataset.RetinopathyDataset(df=test_df, categorical_partitition=True,
                                                cat_labels_to_include=TEST_CAT_LABELS, transforms=train_transform)

print("Length of Concat df: ", len(concat_df))
print("Length of Training dataset: ", train_dataset.__len__())
print("Length of Validation dataset: ", val_dataset.__len__())
print("Length of Test dataset: ", test_dataset.__len__())

#Creating DataLoader
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)

dm = retinopathy_dataset.LightningRetinopathyDataset(train_dataset, BATCH_SIZE)

classifier = retinopathy_model.RetinopathyClassificationModel(encoder=ENCODER, pretrained=True, 
                                                            num_classes=NUM_CLASSES, lr_scheduler=LR_SCHEDULING
                                                            )

cb_early_stopping = EarlyStopping(monitor='train_loss', patience=5, mode='min')
cb_rich_progressbar = RichProgressBar()
#cb_print_table_metrics = PrintTableMetricsCallback()
cb_model_checkpoint = ModelCheckpoint(dirpath=os.path.join(SAVE_DIR, EXPERIMENT_NAME), monitor='val_loss', save_weights_only=True)

#callbacks = [cb_early_stopping, cb_rich_progressbar, cb_print_table_metrics]
callbacks = [cb_early_stopping, cb_rich_progressbar, cb_model_checkpoint]

if(LOG_MODEL):

    #Provide a deault_root_dir to save the model in case model logging is True
    trainer = pl.Trainer(gpus=GPUS, 
                        max_epochs=EPOCHS, 
                        logger=wandb_logger,
                        default_root_dir=os.path.join(SAVE_DIR, EXPERIMENT_NAME),
                        auto_lr_find=AUTO_LR_FIND,
                        callbacks=callbacks)
else:

    #Skip providing a deault_root_dir to save the model in case model logging is False
    trainer = pl.Trainer(gpus=GPUS, 
                        max_epochs=EPOCHS, 
                        logger=wandb_logger,
                        auto_lr_find=AUTO_LR_FIND,
                        callbacks=callbacks)
                        

#Finding the optimal learning rate for model training
if(AUTO_LR_FIND):
    print("~~~~ Finding optimal learning rate before training the model ~~~~~")
    lr_finder = trainer.tuner.lr_find(classifier, dm)
    new_lr = lr_finder.suggestion()
    print("New suggested learning rate is: ", new_lr)
    classifier.hparams.lr = new_lr
else:
    print("~~~~ Using the learning rate provided in the config file ~~~~~")
    classifier.hparams.lr = LR

trainer.fit(classifier, dm)

if(RUN_EVAL_ON == 'test'):
    trainer.test(classifier, dataloaders=test_loader)
elif(RUN_EVAL_ON == 'validation'):
    trainer.test(classifier, dataloaders=val_loader)
