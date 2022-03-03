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

import yaml
from pprint import pprint

import logging
logging.getLogger("lightning").setLevel(logging.ERROR)

# =============================================================================================== # 

with open('/home/raman/MLP-RIQA/src/config_val.yaml') as file:
    yaml_data = yaml.safe_load(file)

pprint(yaml_data)

#################### DEFINE CONSTANTS ####################

DATASET = yaml_data['dataset']
CHECKPOINT_PATH = yaml_data['checkpoint_path']
NUM_CLASSES = yaml_data['num_classes']
ENCODER = yaml_data['encoder']

##########################################################

transforms = T.Compose([
    T.ToPILImage(),
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

backbone = retinopathy_model.RetinopathyClassificationModel(encoder=ENCODER, pretrained=True, 
                                                            num_classes=NUM_CLASSES, lr_scheduler='none'
                                                            )

classifier = backbone.load_from_checkpoint(checkpoint_path=CHECKPOINT_PATH)

print(classifier)

#print('eval file')