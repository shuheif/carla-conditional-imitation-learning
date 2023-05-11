from collections import OrderedDict
import torch
from torch import nn

N_COMMANDS = 8
N_CONTROL_OUTPUTS = 2

class CILModel(nn.Module):
    def __init__(self):
        """
        Implementation of the network layers.
        The image size of the input observations is 200x86 pixels.
        """
        super().__init__()
        self.conv_layers = self.get_conv_layers()
        self.flatten = nn.Flatten()
        self.fully_connected_layers = self.get_fully_connected_layers()
        self.branches = nn.ModuleList([self.branched_network() for _ in range(N_COMMANDS)])

    def get_conv_layers(self):
        """
        Returns a sequence of convolutional layers with batch normalization, dropout and ReLU activation.
        """
        conv_dict = OrderedDict([])
        conv_dict.update(self.conv_block(kernel_size=5, stride=2, in_channels=3, out_channels=32, index=1))
        conv_dict.update(self.conv_block(kernel_size=3, stride=1, in_channels=32, out_channels=32, index=2))
        conv_dict.update(self.conv_block(kernel_size=3, stride=2, in_channels=32, out_channels=64, index=3))
        conv_dict.update(self.conv_block(kernel_size=3, stride=1, in_channels=64, out_channels=64, index=4))
        conv_dict.update(self.conv_block(kernel_size=3, stride=2, in_channels=64, out_channels=128, index=5))
        conv_dict.update(self.conv_block(kernel_size=3, stride=1, in_channels=128, out_channels=128, index=6))
        conv_dict.update(self.conv_block(kernel_size=3, stride=1, in_channels=128, out_channels=256, index=7))
        conv_dict.update(self.conv_block(kernel_size=3, stride=1, in_channels=256, out_channels=256, index=8))
        return nn.Sequential(conv_dict)

    def get_fully_connected_layers(self):
        """
        Returns a sequence of fully connected layers with dropout and ReLU activation.
        """
        fc_dict = OrderedDict([])
        fc_dict.update(self.fully_connected_block(in_features=8192, out_features=512, index=1))
        fc_dict.update(self.fully_connected_block(in_features=512, out_features=512, index=2))
        return nn.Sequential(fc_dict)

    def conv_block(self, kernel_size, stride, in_channels, out_channels, index):
        """
        Returns a convolutional layer with batch normalization, dropout and ReLU activation.
        """
        return OrderedDict([
            (f'conv{index}', nn.Conv2d(in_channels, out_channels, kernel_size, stride)),
            (f'bn{index}', nn.BatchNorm2d(out_channels)),
            (f'cv_dout{index}', nn.Dropout(0.2)),
            (f'cv_relu{index}', nn.ReLU()),
        ])

    def fully_connected_block(self, in_features, out_features, index):
        """
        Returns a fully connected layer with dropout and ReLU activation.
        """
        return OrderedDict([
            (f'linear{index}', nn.Linear(in_features, out_features)),
            (f'lionear_dout{index}', nn.Dropout(0.5)),
            (f'linear_relu{index}', nn.ReLU()),
        ])
    
    def branched_network(self):
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, N_CONTROL_OUTPUTS),
            nn.Tanh(),
        )

    def select_branch(self, branches, one_hot):
        shape = branches.size()
        for i, s in enumerate(shape[2:]):
            one_hot = torch.stack([one_hot for _ in range(s)], dim=i+2)
        return torch.sum(one_hot * branches, dim=1)

    def forward(self, image, command):
        """
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, height, width, channel)
        return         torch.Tensor of size (batch_size, N_CONTROL_OUTPUTS)
        """
        convoluted_image = self.conv_layers(image)
        flattened = self.flatten(convoluted_image)
        image_network_output = self.fully_connected_layers(flattened)
        branch_outputs = [branch(image_network_output) for branch in self.branches]
        branch_outputs = torch.stack(branch_outputs, dim=1)
        control_outputs = self.select_branch(branch_outputs, command)
        return control_outputs


if __name__ == '__main__':
    model = CILModel()
    print(model)
