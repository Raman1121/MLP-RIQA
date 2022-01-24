from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.io import read_image

class RetinopathyDataset(Dataset):
    def __init__(self, train_df, transforms=None, threshold=None):
        self.train_df = train_df
        self.transforms = transforms
        self.threshold = threshold

        if(self.threshold):
            self.train_df = self.train_df[self.train_df['score']>=self.threshold].reset_index(drop=True)

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        image = read_image(self.train_df['image'][idx])
        label = self.train_df['level'][idx]

        if(self.transforms):
            image = self.transforms(image)

        return image, label