import torch
from dataset import retinopathy_dataset
from model import retinopathy_model

from model import retinopathy_model
from torch.utils.data import DataLoader
from torchvision import transforms as T

import pandas as pd
import matplotlib.pyplot as plt
import pytorch_lightning as pl


import argparse
import yaml

# parser = argparse.ArgumentParser(description='Hyper-parameters management')

# parser.add_argument('--batch_size', type=int, default=4, help='Batch Size')
# parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
# parser.add_argument('--encoder', type=str, default='resnet18', help='Encoder model')
# parser.add_argument('--num_classes', type=int, default=5, help='Number of target classes')
# parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate')
# parser.add_argument('--gpus', type=int, default=0, help='Number of GPUs')
# parser.add_argument('--train_df_path', type=str, default='/home/raman/MLP-RIQA/Train_set_RIQA_DR_Labels.csv', help='Path to read the training dataset file')
# parser.add_argument('--cat_labels', type=list, default=['Good', 'Usable', 'Reject'], help='Categorical Quality labels to include in the training set')

# args = parser.parse_args()

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
trainer.fit(classifier, train_loader)  # train_loader

