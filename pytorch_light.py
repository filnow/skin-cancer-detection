import gc
import torch
import lightning as L

from torch import nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader

#NOTE: make it a LightningDataModule
def load_data():
    transformer = transforms.Compose([
        transforms.Resize(size = (IMG_SIZE, IMG_SIZE), antialias = True),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

    testTransformer = transforms.Compose([
        transforms.Resize(size = (IMG_SIZE, IMG_SIZE), antialias = True),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

    trainData = datasets.ImageFolder(root = "./train", transform = transformer)
    testSet = datasets.ImageFolder(root = "./test", transform = testTransformer)

    trainSet, valSet = torch.utils.data.random_split(trainData,\
         [int(0.8 * len(trainData)), len(trainData) - int(0.8 * len(trainData))])
    
    trainLoader = DataLoader(trainSet, batch_size = BATCH_SIZE, shuffle=True)
    valLoader = DataLoader(valSet, batch_size=BATCH_SIZE, shuffle=True)
    testLoader = DataLoader(testSet, batch_size = BATCH_SIZE, shuffle = False)


if __name__ == '__main__':
    IMG_SIZE = 224
    BATCH_SIZE = 32
    IMG_SHOW_NUM = 6
    EPOCHS = 20
    LEARNING_RATE = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")