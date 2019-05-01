import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        '''此模型参考网址：https://medium.com/@vivekvscool/image-classification-cnn-with-pytorch-5b2cb9ef9476'''
        super(CNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(8)  # Batch normalization
        self.relu = nn.ReLU()  # RELU Activation
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # Maxpooling reduces the size by kernel size. 64/2 = 32

        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)  # Size now is 32/2 = 16

        # Flatten the feature maps. You have 32 feature mapsfrom cnn2. Each of the feature is of size 16x16 --> 32*16*16 = 8192
        self.fc1 = nn.Linear(in_features=8192,
                             out_features=4000)  # Flattened image is fed into linear NN and reduced to half size
        self.droput = nn.Dropout(p=0.5)  # Dropout used to reduce overfitting
        self.fc2 = nn.Linear(in_features=4000, out_features=2000)
        self.droput = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=2000, out_features=500)
        self.droput = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(in_features=500, out_features=50)
        self.droput = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(in_features=50,
                             out_features=2)  # Since there were so many features, I decided to use 45 layers to get output layers. You can increase the kernels in Maxpooling to reduce image further and reduce number of hidden linear layers.

    def forward(self, x):
        out = self.cnn1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        # Flattening is done here with .view() -> (batch_size, 32*16*16) = (100, 8192)
        out = out.view(-1, 8192)  # -1 will automatically update the batchsize as 100; 8192 flattens 32,16,16
        # Then we forward through our fully connected layer
        out = self.fc1(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc5(out)
        return out
