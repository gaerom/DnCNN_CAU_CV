import torch
import torch.nn as nn

# class
class DnCNN(nn.Module):
    def __init__(self, num_of_layers=17, num_of_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        '''
        self
        num_of_channels
        num_of_layers

        make sequential layers
        '''
        super(DnCNN, self).__init__()
        padding = 1
        eps = 0.0001
        momentum = 0.95
        # hyper parameter
        self.num_of_layers = num_of_layers
        self.use_bnorm = use_bnorm
        self.input_conv = nn.Conv2d(image_channels, num_of_channels, kernel_size = kernel_size, padding = 'same')
        self.conv = nn.Conv2d(num_of_channels, num_of_channels, kernel_size = kernel_size, padding = 'same')
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(num_of_channels, eps = eps, momentum=momentum)
        self.output_conv = nn.Conv2d(num_of_channels, 1, kernel_size = kernel_size, padding = 'same')

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        '''
        self
        x

        return: x-y
        '''
        # MODEL Architecture
        y = self.input_conv(x)
        y = self.relu(y)
        for d in range(self.num_of_layers-2):
                y = self.conv(y)
                if self.use_bnorm:
                    y = self.bn(y)
                y = self.relu(y)
        y = self.output_conv(y)
        return x - y