import torch
from dataset import retinopathy_dataset
from model import retinopathy_model

from model import retinopathy_model
from torch.utils.data import DataLoader
from torchvision import transforms as T

import pandas as pd
import matplotlib.pyplot as plt
import pytorch_lightning as pl

import yaml

with open('config_train_DR.yaml') as file:
    yaml_data = yaml.safe_load(file)


train_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224,224)),
    T.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_df = pd.read_csv(yaml_data['train']['train_df_path'])
print("Length of training df: ", len(train_df))
dataset = retinopathy_dataset.RetinopathyDataset(train_df=train_df, categorical_partitition=True,
                                                cat_labels_to_include=yaml_data['train']['cat_labels'], transforms=train_transform)

train_loader = DataLoader(dataset, batch_size=yaml_data['train']['batch_size'], shuffle=True)

classifier = retinopathy_model.RetinopathyClassificationModel(encoder=yaml_data['model']['encoder'], pretrained=True, 
                                                            num_classes=yaml_data['train']['num_classes'], lr=yaml_data['train']['lr'])

trainer = pl.Trainer(progress_bar_refresh_rate=20, gpus=yaml_data['train']['gpus'], max_epochs=yaml_data['train']['epochs'])  
trainer.fit(classifier, train_loader)  

