import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) for binary classification of spectrogram images.

    The architecture consists of:
    - Three convolutional layers with batch normalization, max pooling, and dropout.
    - Three fully connected layers with dropout.
    - A final fully connected layer for binary classification.
    """
    def __init__(self):
        """
        Initializes the CNN model by defining its layers.

        Layers include:
        - Convolutional layers with ReLU activation, batch normalization, and dropout.
        - Max pooling layers to reduce spatial dimensions.
        - Fully connected layers for classification.
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.bn1 = nn.BatchNorm2d(32)
        self.d1 = nn.Dropout(0.3)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 5)
        self.bn2 = nn.BatchNorm2d(64)
        self.d2 = nn.Dropout(0.5)

        self.conv3 = nn.Conv2d(64, 128, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.d3 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(128*2*2, 64)
        self.d5 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(64, 32)
        self.d6 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(32, 16)

        self.fc4 = nn.Linear(16, 2)

    def forward(self, x):
        """
        Defines the forward pass of the CNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 2), representing class scores.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bn1(x)
        x = self.d1(x)

        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn2(x)
        x = self.d2(x)

        x = self.pool(F.relu(self.conv3(x)))
        x = self.bn3(x)
        x = self.d3(x)
        
        x = x.view(-1, 128*2*2)

        x = F.relu(self.fc1(x))
        x = self.d5(x)

        x = F.relu(self.fc2(x))
        x = self.d6(x)

        x = F.relu(self.fc3(x))

        x = self.fc4(x)
        return x
