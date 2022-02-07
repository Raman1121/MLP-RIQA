from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.io import read_image

class RetinopathyDataset(Dataset):
    def __init__(self, train_df, transforms=None, threshold=None, categorical_partitition=True, cat_labels_to_include=['Good', 'Usable', 'Bad']):
        """
            Parameters
            ----------
            train_df : pandas.core.frame.DataFrame
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

        self.train_df = train_df
        self.transforms = transforms
        self.threshold = threshold
        self.categorical_partitition = categorical_partitition
        self.cat_labels = cat_labels_to_include

        if(self.categorical_partitition == True):
            #This means we have categorical labels for image quality ['Good', 'Usable', 'Bad'] instead of continuous labels
            self.threshold = None   #Continuous labels threshold does not matter in this case

            if(self.cat_labels == None):
                raise AssertionError("Categorical labels should be provided when 'categorical partition' is set to True")
            else:
                self.train_df = self.train_df[self.train_df['quality'].isin(self.cat_labels).reset_index(drop=True)]

        else:
            if(self.threshold == None):
                raise AssertionError("Threshold should be provided when 'categorical_partitition' is false.")
                
            self.train_df = self.train_df[self.train_df['score']>=self.threshold].reset_index(drop=True)

    def __len__(self):
        """
            Returns
            -------

            Number of samples in our dataset.
        """
        return len(self.train_df)

    def __getitem__(self, idx):
        """
            Parameters
            ----------
            idx: index to identify a sample in the dataset

            Returns
            -------
            An image and a label from the dataset based on the given index idx.
        """
        image = read_image(self.train_df['image'][idx])
        label = self.train_df['level'][idx]

        if(self.transforms):
            image = self.transforms(image)

        return image, label