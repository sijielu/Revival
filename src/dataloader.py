# Revival: Convert Minecraft to Real World
# CS 175: Project in Artificial Intelligence (Spring 2019)
#
# dataloader.py
#

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

BATCH_SIZE = 1


class RevivalDataLoader(object):
    @staticmethod
    def create_dataset(path):
        # transform = transforms.Compose([transforms.Resize((256, 256)),
        transform = transforms.Compose([transforms.Resize((128, 128)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = datasets.ImageFolder(path, transform)
        return dataset

    def __init__(self, pathX, pathY, is_train=True):
        self.datasetX = RevivalDataLoader.create_dataset(pathX)
        self.datasetY = RevivalDataLoader.create_dataset(pathY)
        self.data_loaderX = DataLoader(self.datasetX, BATCH_SIZE, is_train)
        self.data_loaderY = DataLoader(self.datasetY, BATCH_SIZE, is_train)

    # def __len__(self):
    #     return min(len(self.datasetX), len(self.datasetY))

    def __iter__(self):
        for dataX, dataY in zip(self.data_loaderX, self.data_loaderY):
            yield dict(X=dataX, Y=dataY)
