# importing
import torch
import torch.nn as nn

# class
class DnCNN(nn.Module):
    def __init__(self, num_of_channels, num_of_layers):
        '''
        self
        num_of_channels
        num_of_layers

        make sequential layers
        '''
        super().__init__()
        kernel_size = 3
        # hyper parameter
        self.depth = 17
        self.is_used_bn = True
        eps = 0.0001
        momentum = 0.95

        self.input_conv = nn.Conv2d(1, num_of_channels, kernel_size = kernel_size, padding = 'same')
        self.conv = nn.Conv2d(num_of_channels, num_of_channels, kernel_size = kernel_size, padding = 'same')
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(num_of_channels, eps = eps, momentum=momentum)
        self.output_conv = nn.Conv2d(num_of_channels, 1, kernel_size = kernel_size, padding = 'same')
        # initialize_weights
    
    def _initialize_weights(self):
        pass
    
    def forward(self, x):
        '''
        self
        x

        return: x-y
        '''
        # MODEL Architecture
        y = self.input_conv(x)
        y = self.relu(y)
        for d in range(self.depth-2):
                y = self.conv(y)
                if self.is_used_bn:
                    y = self.bn(y)
                y = self.relu(y)
        y = self.output_conv(y)
        return x - y