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

matplotlib.use('TkAgg')


def main(args: argparse.Namespace, activeloop: bool = True) -> None:
    # Load the train and test data set

    """"
    Augmentation should be applied to botyh train and test datasets. To apply, simply change the flag:
    dataset = ImageDataset(x='path/to/images.npy', y='path/to/labels.npy', augment=True)
    """
    train_dataset = ImageDataset(Path("../data/X_train.npy"),
                                 Path("../data/Y_train.npy"))
    test_dataset = ImageDataset(Path("../data/X_test.npy"),
                                Path("../data/Y_test.npy"))

    # Load the Neural Net. NOTE: set number of distinct labels here
    model = Net(n_classes=6)

    # Initialize optimizer(s) and loss function(s)
    # ORIGINAL    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.1)
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.1)
    loss_function = nn.CrossEntropyLoss()

    # fetch epoch and batch count from arguments
    n_epochs = args.nb_epochs
    batch_size = args.batch_size

    # IMPORTANT! Set this to True to see actual errors regarding
    # the structure of your model (GPU acceleration hides them)!
    # Also make sure you set this to False again for actual model training
    # as training your model with GPU-acceleration (CUDA/MPS) is much faster.
    DEBUG = False

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

    # Let's now train and test our model for multiple epochs:
    train_sampler = BatchSampler(
        batch_size=batch_size, dataset=train_dataset, balanced=args.balanced_batches
    )
    test_sampler = BatchSampler(
        batch_size=100, dataset=test_dataset, balanced=args.balanced_batches
    )

    mean_losses_train: List[torch.Tensor] = []
    mean_losses_test: List[torch.Tensor] = []
    accuracies_train, precisions_train, recalls_train = [], [], []
    accuracies_test, precisions_test, recalls_test = [], [], []
    roc_auc_data = []

    for e in range(n_epochs):
        if activeloop:
            # Training:

            train_losses, train_acc, train_prec, train_rec = train_model(model, train_sampler, optimizer, loss_function,
                                                                         device)
            # Calculating and printing statistics:
            accuracies_train.append(train_acc)
            precisions_train.append(train_prec)
            recalls_train.append(train_rec)
            mean_loss = sum(train_losses) / len(train_losses)
            mean_losses_train.append(mean_loss)
            print(
                f"\nEpoch {e + 1} training done:\n"
                f"Training Metrics:\n"
                f"Loss: {mean_loss}\n"
                f"🎯 Accuracy: {train_acc}\n"
                f"🎯 Precision: {train_prec}\n"
                f"🎯 Recall: {train_rec}\n"
            )

            # Testing:
            test_losses, test_acc, test_prec, test_rec, roc_auc_dict = test_model(model, test_sampler, loss_function,
                                                                                  device)
            # Getting ROC data
            print(f"ROC AUC dict for epoch {e + 1}: {roc_auc_dict}")
            roc_auc_data.append(roc_auc_dict)
            # # Calculating and printing statistics:
            accuracies_test.append(test_acc)
            precisions_test.append(test_prec)
            recalls_test.append(test_rec)
            mean_loss = sum(test_losses) / len(test_losses)
            mean_losses_test.append(mean_loss)
            print(
                f"\nEpoch {e + 1} testing done:\n"
                f"Testing Metrics:\n"
                f"Loss: {mean_loss}\n"
                f"🎯 Accuracy: {test_acc}\n"
                f"🎯 Precision: {test_prec}\n"
                f"🎯 Recall: {test_rec}\n"
            )

            # Plotting during training
            plotext.clf()
            plotext.scatter(mean_losses_train, label="Тrain Loss")
            plotext.scatter(mean_losses_test, label="Test Loss")

            plotext.title("Train and test loss")

            plotext.xticks([i for i in range(len(mean_losses_train) + 1)])

            plotext.show()

    # retrieve current time to label artifacts
    now = datetime.now()

    # finding the minimum mean loss for training and testing sets
    min_mean_loss_train = min([x.item() for x in mean_losses_train]) if mean_losses_train else None
    min_mean_loss_test = min([x.item() for x in mean_losses_test]) if mean_losses_test else None

    # Saving the model with additional information
    model_info = {
        "state_dict": model.state_dict(),
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "mean_losses_train": [x.item() for x in mean_losses_train],  # Convert tensors to numbers
        "mean_losses_test": [x.item() for x in mean_losses_test],
        "min_mean_loss_train": min_mean_loss_train,  # Add minimum mean loss for training
        "min_mean_loss_test": min_mean_loss_test,  # Add minimum mean loss for testing
    }

    # Check if model_weights/ subdir exists
    if not Path("model_weights/").exists():
        os.mkdir(Path("model_weights/"))

    # Construct the filename using current date and time
    filename = f"model_weights/model_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.pt"  # Use .pt for PyTorch models

    # Save the comprehensive model information
    torch.save(model_info, filename)

    # create plot of losses and accuracies
    figure(figsize=(9, 10), dpi=80)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    # Plot train and test losses on the first subplot (ax1)
    ax1.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_train], label="Train Loss", marker='o',
             color="blue")
    ax1.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_test], label="Test Loss", marker='o',
             color="red")

    # Set titles and labels for the first subplot (ax1)
    ax1.set_title("Train and Test Loss Over Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc='upper right')

    # Plot train and test accuracies on the second subplot (ax2)
    ax2.plot(range(1, 1 + n_epochs), accuracies_train, label="Train Accuracy", marker='o', color="blue")
    ax2.plot(range(1, 1 + n_epochs), accuracies_test, label="Test Accuracy", marker='o', color="red")

    # Set titles and labels for the second subplot (ax2)
    ax2.set_title("Train and Test Accuracy Over Epochs")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc='upper right')

    # Plotting and saving ROC Curves
    roc_figure, roc_axes = plt.subplots(figsize=(10, 8))
    for _, label_name in train_test.label_names.items():
        epoch_auc_scores = [roc_auc_data[epoch][label_name] for epoch in range(n_epochs)]
        roc_axes.plot(range(1, n_epochs + 1), epoch_auc_scores, marker='o', label=label_name)

    roc_axes.set_xlabel('Epochs')
    roc_axes.set_ylabel('ROC AUC')
    roc_axes.set_title('ROC AUC per Class Over Epochs')
    roc_axes.legend(loc='lower right')

    # Check if /artifacts/ subdir exists
    if not Path("artifacts/").exists():
        os.mkdir(Path("artifacts/"))

    # Save the losses and accuracy plots in the artifacts folder
    fig.savefig(Path("artifacts") / f"session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")

    # Save the ROC plot in the artifacts folder
    roc_figure.savefig(Path("artifacts") / f"roc_curves_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")


# def perform_grid_search():
#     epoch_range = [5, 10, 15]
#     batch_size_range = [16, 32, 64]
#     best_loss = float('inf')
#     best_config = None
#
#     for epochs in epoch_range:
#         for batch_size in batch_size_range:
#             print(f"Testing configuration: epochs={epochs}, batch_size={batch_size}")
#             # Assuming your args variable is still needed for other configurations
#             args = argparse.Namespace(nb_epochs=epochs, batch_size=batch_size, balanced_batches=True)
#             # Call your modified main function with the current combination
#             # You might need to capture the performance metric from your main function to use here
#             main(args, epochs, batch_size, activeloop=True)
#             # Here, you'd capture the performance metric and update best_loss and best_config accordingly
#
#     print(f"Best configuration: epochs={best_config['epochs']}, batch_size={best_config['batch_size']} with loss {best_loss}")


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
