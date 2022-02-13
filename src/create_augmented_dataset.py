from torchvision import transforms as T
import pandas as pd
from PIL import Image
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
import albumentations as A

#Define Constants
ROTATION_LIMIT = 30

with open('config_train_DR.yaml') as file:
    yaml_data = yaml.safe_load(file)

#################### DEFINE CONSTANTS ####################
TRAIN_DF_PATH = yaml_data['train']['train_df_path']
VALIDATION_SPLIT = yaml_data['train']['validation_split']
SEED = yaml_data['train']['seed']
DATASET_ROOT_PATH = yaml_data['dataset']['root_path']


#Define Albumentation augmentations here
aug_transforms = A.Compose([
                    A.OneOf([
                        A.VerticalFlip(p=1),
                        A.HorizontalFlip(p=1),
                        A.Rotate(limit=ROTATION_LIMIT, p=1),
                        A.GaussianBlur(p=1)
                    ], p=1)
                ])

main_df = pd.read_csv(TRAIN_DF_PATH)

# Creating training and validation splits
train_df, val_df = train_test_split(main_df, test_size=VALIDATION_SPLIT,
                                    random_state=SEED)

#Augment only the good images
train_df_good = train_df.loc[train_df['quality'] == 'Good'].reset_index(drop=True)

val_df = val_df.reset_index(drop=True)

'''
Steps to augment the dataset:
    1. Iterate through train_df_good
    2. Select one of the augmentations randomly
    3. Augment the image
    4. Save the augmented copy inside data/diabetic-retinopathy-detection/augmented_data/
'''

for i in range(len(train_df_good)):
    _name = train_df_good['image'][i]
    _path = str(DATASET_ROOT_PATH+'final_train/train/' + _name)
    print(_path)
    pillow_image = Image.open(_path)
    image = np.array(pillow_image)
    augmented_image = aug_transforms(image=image)['image']

    _save_path = DATASET_ROOT_PATH+'augmented_data/'+'augmented_'+_name

    aug_PIL = Image.fromarray(augmented_image)
    aug_PIL.save(_save_path)



