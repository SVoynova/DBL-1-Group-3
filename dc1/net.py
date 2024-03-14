import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np


# class Net(nn.Module):
#     def __init__(self, n_classes: int) -> None:
#         super(Net, self).__init__()
#
#         self.cnn_layers = nn.Sequential(
#             # First 2D convolution layer
#             nn.Conv2d(1, 64, kernel_size=4, stride=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=4),
#             torch.nn.Dropout(p=0.5, inplace=True),
#
#             # Second 2D convolution layer
#             nn.Conv2d(64, 32, kernel_size=4, stride=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3),
#             torch.nn.Dropout(p=0.25, inplace=True),
#
#             # Third 2D convolution layer
#             nn.Conv2d(32, 16, kernel_size=4, stride=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),
#             torch.nn.Dropout(p=0.125, inplace=True),
#
#             nn.AdaptiveAvgPool2d((1, 1))
#         )
#
#         self.linear_layers = nn.Sequential(
#             nn.Flatten(),  # Flatten the output of adaptive avg pooling
#             nn.Linear(16, 256),  # Adjust input size based on the actual output of CNN
#             nn.Linear(256, n_classes),
#             nn.Softmax(dim=1)
#         )
#
#     # Defining the forward pass
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         # x = self.cnn_layers(x)
#         # # After our convolutional layers which are 2D, we need to flatten our
#         # # input to be 1 dimensional, as the linear layers require this.
#         # x_prediction = x.view(x.size(0), -1)
#         # x = self.linear_layers(x_prediction)
#         # return x
#         x = self.cnn_layers(x)
#         x = self.linear_layers(x)
#         return x  # Directly return logits for compatibility with CrossEntropyLoss



class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.forward_output = None

        target_layer.register_forward_hook(self.save_forward_output)
        target_layer.register_full_backward_hook(self.save_gradients)

    def save_forward_output(self, module, input, output):
        self.forward_output = output

    def save_gradients(self, module, input_grad, output_grad):
        self.gradients = output_grad[0]

    def generate_heatmap(self, input_image, target_class):
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        if target_class is None:
            target_class = np.argmax(output.cpu().data.numpy())

        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        output.backward(gradient=one_hot_output, retain_graph=True)
        # Convert gradients to positive values and take the maximum
        gradients_positive = torch.abs(self.gradients)
        pooled_gradients = torch.mean(gradients_positive, dim=[0, 2, 3])
        # Weight the channels by corresponding gradients
        for i in range(gradients_positive.size()[1]):
            self.forward_output.data[0, i, :, :] *= pooled_gradients[i]
        # Average the channels of the activations
        heatmap = torch.mean(self.forward_output, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu().detach().numpy(), 0)
        # Normalize the heatmap
        heatmap /= np.max(heatmap)
        return heatmap

    def visualize(self, input_image, target_class=None):
        heatmap = self.generate_heatmap(input_image, target_class)
        plt.matshow(heatmap)
        plt.show()


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=1):
        super(ResidualBlock, self).__init__()
        mid_channels = out_channels // expansion
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Net(nn.Module):
    def __init__(self, n_classes: int):
        super(Net, self).__init__()

        self.init_conv = nn.Sequential(
#            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Increased depth and varied width
        self.layer1 = nn.Sequential(ResidualBlock(64, 64), ResidualBlock(64, 64))
        self.layer2 = nn.Sequential(ResidualBlock(64, 128, stride=2), ResidualBlock(128, 128))
        self.layer3 = nn.Sequential(ResidualBlock(128, 256, stride=2), ResidualBlock(256, 256))
        self.layer4 = nn.Sequential(ResidualBlock(256, 512, stride=2), ResidualBlock(512, 512))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_layers = nn.Sequential(
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)

        return x