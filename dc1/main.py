# Custom imports
from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from dc1.net import Net
from dc1.train_test import train_model, test_model
import train_test

# Torch imports

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary  # type: ignore
import torch
import torch.nn as nn

# Other imports
import matplotlib
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.pyplot import figure
import argparse
import plotext  # type: ignore
from datetime import datetime
from pathlib import Path
from typing import List
import os

# Utility class for the evaluation metric things
from evaluationMetricUtility import EvaluationMettricsLogger
matplotlib.use('TkAgg')


def main(args: argparse.Namespace, activeloop: bool = True) -> None:

    # Load the train and test datasets with optional augmentation // Flag = True of data augmentation
    train_dataset = ImageDataset(Path("/Users/ss.voynova/DBL-1-Group-3/data/X_train.npy"),
                                 Path("/Users/ss.voynova/DBL-1-Group-3/data/Y_train.npy"))
    test_dataset = ImageDataset(Path("/Users/ss.voynova/DBL-1-Group-3/data/X_test.npy"),
                                Path("/Users/ss.voynova/DBL-1-Group-3/data/Y_test.npy"))

    # Initialize the Neural Net with the number of distinct labels
    model = Net(n_classes=6)

    # Initialize optimizer and loss function - original params: lr=0.001, momentum=0.1
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.1)
    loss_function = nn.CrossEntropyLoss()

    # Fetch epoch and batch count from arguments
    n_epochs = args.nb_epochs
    batch_size = args.batch_size

    # IMPORTANT! Set this to True to see actual errors regarding
    # the structure of your model (GPU acceleration hides them)!
    # Also make sure you set this to False again for actual model training
    # as training your model with GPU-acceleration (CUDA/MPS) is much faster.
    DEBUG = False

    # ! Setup model device (CPU, CUDA, MPS)
    model, device = setup_model_device(model, DEBUG)

    # Initialize batch samplers for training and testing
    train_sampler = BatchSampler(
        batch_size=batch_size, dataset=train_dataset, balanced=args.balanced_batches
    )
    test_sampler = BatchSampler(
        batch_size=100, dataset=test_dataset, balanced=args.balanced_batches
    )

    # Instantiate EvaluationMetricsLogger for logging and plotting metrics
    metrics_logger = EvaluationMettricsLogger()

    # Loop over epochs to train and test the model, logging the metrics
    for e in range(n_epochs):
        if activeloop:
            metrics_logger.log_training_epochs(e, model, train_sampler, optimizer, loss_function, device)
            metrics_logger.log_testing_epochs(e, model, test_sampler, loss_function, device)

            # Plots the both training and the testing losses of the logged model
            metrics_logger.plot_training_testing_losses()


    if not Path("model_weights/").exists():
        os.mkdir(Path("model_weights/"))

    # Saves the model
    metrics_logger.save_model(model, n_epochs, batch_size)

    # Plot overall loss and accuracy, and ROC curve for model evaluation
    metrics_logger.plot_loss_and_accuracy(n_epochs)
    metrics_logger.plot_roc_curve(n_epochs, train_test)

def setup_model_device(model, DEBUG):
    """
    Move the model to the appropriate device based on availability and debug flag.
    """
    # Moving our model to the right device (CUDA will speed training up significantly!)
    if torch.cuda.is_available() and not DEBUG:
        print("@@@ CUDA device found, enabling CUDA training...")
        device = "cuda"
        model.to(device)
        # Creating a summary of our model and its layers:
        summary(model, (1, 128, 128), device=device)
    elif (
            torch.backends.mps.is_available() and not DEBUG
    ):  # PyTorch supports Apple Silicon GPU's from version 1.12
        print("@@@ Apple silicon device enabled, training with Metal backend...")
        device = "mps"
        model.to(device)
    else:
        print("@@@ No GPU boosting device found, training on CPU...")
        device = "cpu"
        # Creating a summary of our model and its layers:
        summary(model, (1, 128, 128), device=device)
    return model, device

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nb_epochs", help="number of training iterations", default=1, type=int
    )
    parser.add_argument("--batch_size", help="batch_size", default=1, type=int)
    parser.add_argument(
        "--balanced_batches",
        help="whether to balance batches for class labels",
        action='store_true'
    )
    parser.set_defaults(balanced_batches=False)
    args = parser.parse_args()

    main(args)
