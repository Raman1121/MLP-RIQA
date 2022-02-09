import torch
from dataset import retinopathy_dataset
from model import retinopathy_model

from model import retinopathy_model
from torch.utils.data import DataLoader
from torchvision import transforms as T

import pandas as pd
import matplotlib.pyplot as plt
import pytorch_lightning as pl

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


train_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224,224)),
    T.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

print("Length of Main df: ", len(main_df))
print("Length of Training df: ", len(train_df))
print("Length of Validation df: ", len(val_df))
print("Length of Test df: ", len(test_df))

#Creating Datasets
train_dataset = retinopathy_dataset.RetinopathyDataset(df=train_df, categorical_partitition=True,
                                                cat_labels_to_include=yaml_data['train']['cat_labels'], transforms=train_transform)

val_dataset = retinopathy_dataset.RetinopathyDataset(df=val_df, categorical_partitition=True,
                                                cat_labels_to_include=yaml_data['train']['cat_labels'], transforms=train_transform)

test_dataset = retinopathy_dataset.RetinopathyDataset(df=test_df, categorical_partitition=True,
                                                cat_labels_to_include=yaml_data['train']['cat_labels'], transforms=train_transform)

#Creating Dataloaders
# train_loader = DataLoader(train_dataset, batch_size=yaml_data['train']['batch_size'], shuffle=True, num_workers=12)
# val_loader = DataLoader(val_dataset, batch_size=yaml_data['train']['batch_size'], shuffle=False, num_workers=12)
# test_loader = DataLoader(test_dataset, batch_size=yaml_data['train']['batch_size'], shuffle=False, num_workers=12)

dm = retinopathy_dataset.LightningRetinopathyDataset(train_dataset, val_dataset, test_dataset, yaml_data['train']['batch_size'])


classifier = retinopathy_model.RetinopathyClassificationModel(encoder=yaml_data['model']['encoder'], pretrained=True, 
                                                            num_classes=yaml_data['train']['num_classes'], lr=yaml_data['train']['lr'])

trainer = pl.Trainer(gpus=yaml_data['train']['gpus'], max_epochs=yaml_data['train']['epochs'])  
#trainer.fit(classifier, train_loader, val_loader)
trainer.fit(classifier, dm)

trainer.test(dm)

