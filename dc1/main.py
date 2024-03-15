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
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

def main(args: argparse.Namespace, activeloop: bool = True) -> None:

    # Load the train and test datasets. Enable data augmentation by setting the augmentation flag to True.
    # The second 'False' refers to the augmentation flag (set to True to enable augmentation).
    # The first 'False' addresses class imbalance; it should be set to True for training data to handle imbalance,
    # and False for test data. If no 'labels_for_augmentation' is specified but augmentation is enabled,
    # apply augmentation to all classes.
    train_dataset = ImageDataset(Path("../data/X_train.npy"),
                                 Path("../data/Y_train.npy"), False)
    test_dataset = ImageDataset(Path("../data/X_test.npy"),
                                Path("../data/Y_test.npy"), False)

    # Initialize the Neural Net with the number of distinct labels
    model = Net(n_classes=6)

    # Initialize optimizer and loss function - original params: lr=0.001, momentum=0.1
    #optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.1)

    optimizer = AdamW(model.parameters(), lr=0.01)  # AdamW requires a lower LR generally

    # Define a scheduler as before
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    #    loss_function = nn.CrossEntropyLoss()

    #Modified loss function to compensate for class imbalances
#    class_weights = torch.tensor([2.42, 2.63, 2.06, 1.0, 3.74, 4.69], dtype=torch.float)
    class_weights = torch.tensor([2., 2., 2, 1.0, 3, 4], dtype=torch.float)

    # If you're using a GPU, ensure to transfer the weights to the same device as your model and data
    if torch.cuda.is_available():
        class_weights = class_weights.to('cuda')

    loss_function = nn.CrossEntropyLoss(weight=class_weights)

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
            scheduler.step()


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
        "--nb_epochs", help="number of training iterations", default=15, type=int
    )
    parser.add_argument("--batch_size", help="batch_size", default=32, type=int)
    parser.add_argument(
        "--balanced_batches",
        help="whether to balance batches for class labels",
        action='store_true'
    )
    parser.set_defaults(balanced_batches=False)
    args = parser.parse_args()

    main(args)

