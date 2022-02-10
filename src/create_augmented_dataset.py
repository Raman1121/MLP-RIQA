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

#Define Albumentation augmentations here
aug_transforms = A.Compose([
                    A.OneOf([
                        A.VerticalFlip(p=1),
                        A.HorizontalFlip(p=1),
                        A.Rotate(limit=ROTATION_LIMIT, p=1),
                        A.GaussianBlur(p=1)
                    ], p=1)
                ])

main_df = pd.read_csv(yaml_data['train']['train_df_path'])

#main_df['image'] = main_df['image'].apply(lambda x: str(yaml_data['dataset']['root_path']+'final_train/train/'+x))

# Creating training and validation splits
train_df, val_df = train_test_split(main_df, test_size=yaml_data['train']['validation_split'],
                                    random_state=yaml_data['train']['seed'])

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
    _path = str(yaml_data['dataset']['root_path']+'final_train/train/' + _name)
    print(_path)
    pillow_image = Image.open(_path)
    image = np.array(pillow_image)
    augmented_image = aug_transforms(image=image)['image']

    _save_path = yaml_data['dataset']['root_path']+'augmented_data/'+'augmented_'+_name

    #cv2.imwrite(_save_path, augmented_image)
    aug_PIL = Image.fromarray(augmented_image)
    aug_PIL.save(_save_path)



