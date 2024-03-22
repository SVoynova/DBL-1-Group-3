import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np


class MCDropoutAnalysis:
    def __init__(self, epoch_softmax_averages):
        """
        Initializes the analysis class with softmax probabilities.

        :param epoch_softmax_averages: A list of tensors, each representing the average softmax probabilities across MC dropout runs for an epoch.
        """
        self.data = torch.stack(epoch_softmax_averages)

    def plot_probability_distributions(self):
        """Plots the probability distribution for each class using KDE."""
        for i in range(self.data.size(1)):  # Loop through each class
            sns.kdeplot(self.data[:, i].numpy(), label=f'Class {i}')
        plt.title('Probability Distribution per Class across Epochs')
        plt.xlabel('Softmax Probability')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

    def plot_trend_analysis(self):
        """Plots the trend of average softmax probability for each class across epochs."""
        for i in range(self.data.size(1)):  # Loop through each class
            plt.plot(self.data[:, i].numpy(), label=f'Class {i}')
        plt.title('Trend of Average Softmax Probability across Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Average Softmax Probability')
        plt.legend()
        plt.show()

    def print_statistics(self):
        """Prints basic statistics for each class across all epochs."""
        means = torch.mean(self.data, dim=0)
        std_devs = torch.std(self.data, dim=0)
        for i in range(means.size(0)):
            print(f'Class {i}: Mean = {means[i].item():.4f}, Std Dev = {std_devs[i].item():.4f}')