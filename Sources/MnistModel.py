import torch
from torchvision import datasets, transforms
import torch.nn as nn

def ImportData(TestSize):
    #Batch parameters
    BatchSizeTrain = 64
    BatchSizeTest = TestSize

    #Picture normalization
    AllTransforms = transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor()])

    #datasets download
    TrainDataSet = datasets.MNIST(root = './data', train = True, transform = AllTransforms, download = True)
    TestDataset = datasets.MNIST(root = './data', train = False, transform = AllTransforms, download=True)

    print(len(TrainDataSet))
    print(len(TestDataset))

    #The datasets are converted into a loader
    TrainLoader = torch.utils.data.DataLoader(dataset = TrainDataSet, batch_size = BatchSizeTrain, shuffle = False)
    TestLoader = torch.utils.data.DataLoader(dataset = TestDataset,batch_size = BatchSizeTest, shuffle = False)
    
    return TrainLoader, TestLoader

#Model definition
class ConvNeuralNet(nn.Module):

    #Layers initilisation
    def __init__(self, num_classes):

        #Base class called
        super(ConvNeuralNet, self).__init__()

        #3 Convolution layers
        self.Conv1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride = 1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.Conv2 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=128, stride = 1, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=128, stride = 1, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.Conv3 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=256, stride = 1, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=256, stride = 1, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        #3 Linear layers
        self.F1 = nn.Linear(2304, 1024)
        self.Act1 = nn.ReLU()
        self.F2 = nn.Linear(1024, 512)
        self.Act2 = nn.ReLU()
        self.F3 = nn.Linear(512, num_classes)

    #Function called to compute a prediction
    def forward(self, x):

      #Convolutional layers
      Out = self.Conv1(x)
      Out = self.Conv2(Out)
      Out = self.Conv3(Out)

      #Pictures are converted into vectors
      Out = Out.reshape(Out.size(0), -1)

      #Linear layers
      Out = self.F1(Out)
      Out = self.Act1(Out)
      Out = self.F2(Out)
      Out = self.Act2(Out)
      Out = self.F3(Out)

      return Out