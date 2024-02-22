import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

def ImportData(TestSize):
    #Batch parameters
    BatchSizeTrain = 64
    BatchSizeTest = TestSize

    #Picture normalization
    AllTransforms =   transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])

    #datasets download
    TrainDataSet = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=AllTransforms)
    TestDataset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=AllTransforms)

    print(len(TrainDataSet))
    print(len(TestDataset))

    #The datasets are converted into a loader
    TrainLoader = torch.utils.data.DataLoader(dataset = TrainDataSet, batch_size = BatchSizeTrain, shuffle = False)
    TestLoader = torch.utils.data.DataLoader(dataset = TestDataset,batch_size = BatchSizeTest, shuffle = False)
    
    return TrainLoader, TestLoader

class Cifar10CNN(nn.Module):
    def __init__(self):
        super(Cifar10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4) 
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x