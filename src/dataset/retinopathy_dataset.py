from cgi import test
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.io import read_image
import pytorch_lightning as pl

import yaml

with open('config_train_DR.yaml') as file:
    yaml_data = yaml.safe_load(file)

#yaml_data = read_config_file(verbose=True)

class RetinopathyDataset(Dataset):
    def __init__(self, df, transforms=None, threshold=None, 
                 categorical_partitition=True, cat_labels_to_include=yaml_data['train']['cat_labels']):
        """
            Parameters
            ----------
            df : pandas.core.frame.DataFrame
                Dataframe containing image paths ['image'], retinopathy level ['level'], and image quality scores ['score']
            transforms : torchvision.transforms.transforms.Compose, default: None
                A list of torchvision transformers to be applied to the training images.
            threshold : float, default: None
                The quality threshold below images would be discarded from the training set.
            categorical_partition : bool, default: True
                A variable to denote if RIQA labels are categorical in nature.
            cat_labels_to_include : list, default: ['Good', 'Usable', 'Bad']
                A list of categorical labels to be included in our dataset
        """

        self.df = df
        self.transforms = transforms
        self.threshold = threshold
        self.categorical_partitition = categorical_partitition
        self.cat_labels = cat_labels_to_include

        if(self.categorical_partitition == True):
            #This means we have categorical labels for image quality ['Good', 'Usable', 'Reject'] instead of continuous labels
            self.threshold = None   #Continuous labels threshold does not matter in this case

            if(self.cat_labels == None):
                raise AssertionError("Categorical labels should be provided when 'categorical partition' is set to True")
            else:
                #self.df = self.df[self.df['quality'].isin(self.cat_labels).reset_index(drop=True)]
                self.df = self.df.loc[self.df['quality'].isin(self.cat_labels)].reset_index(drop=True)

        else:
            if(self.threshold == None):
                raise AssertionError("Threshold should be provided when 'categorical_partitition' is false.")
                
            #self.df = self.df[self.df['score']>=self.threshold].reset_index(drop=True)
            self.df = self.df.loc[self.df['score'] >= self.threshold].reset_index(drop=True)

    def __len__(self):
        """
            Returns
            -------

            Number of samples in our dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
            Parameters
            ----------
            idx: index to identify a sample in the dataset

            Returns
            -------
            An image and a label from the dataset based on the given index idx.
        """
        image = read_image(self.df['image'][idx])
        label = self.df['level'][idx]

        if(self.transforms):
            image = self.transforms(image)

        return image, label


class LightningRetinopathyDataset(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size=32):
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=12)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=12)
    