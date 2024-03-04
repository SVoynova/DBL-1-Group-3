# Custom imports
from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from dc1.net import Net
from dc1.train_test import train_model, test_model

# Torch imports

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary  # type: ignore
import torch
import torch.nn as nn

# Other imports
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.pyplot import figure
import argparse
import plotext  # type: ignore
from datetime import datetime
from pathlib import Path
from typing import List
import os


def main(args: argparse.Namespace, activeloop: bool = True) -> None:

    # Load the train and test data set
    train_dataset = ImageDataset(Path("../data/X_train.npy"),
                                 Path("../data/Y_train.npy"))
    test_dataset = ImageDataset(Path("../data/X_test.npy"),
                                Path("../data/Y_test.npy"))

    # Load the Neural Net. NOTE: set number of distinct labels here
    model = Net(n_classes=6)

    # Initialize optimizer(s) and loss function(s)
#ORIGINAL    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.1)
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

    for e in range(n_epochs):
        if activeloop:
            # Training:
            losses = train_model(model, train_sampler, optimizer, loss_function, device)
            # Calculating and printing statistics:
            mean_loss = sum(losses) / len(losses)
            mean_losses_train.append(mean_loss)
            print(f"\nEpoch {e + 1} training done, loss on train set: {mean_loss}\n")

            # Testing:
            losses = test_model(model, test_sampler, loss_function, device)

            # # Calculating and printing statistics:
            mean_loss = sum(losses) / len(losses)
            mean_losses_test.append(mean_loss)
            print(f"\nEpoch {e + 1} testing done, loss on test set: {mean_loss}\n")

            ### Plotting during training
            plotext.clf()
            plotext.scatter(mean_losses_train, label="train")
            plotext.scatter(mean_losses_test, label="test")
            plotext.title("Train and test loss")

            plotext.xticks([i for i in range(len(mean_losses_train) + 1)])

            plotext.show()

    # retrieve current time to label artifacts
    now = datetime.now()

    #finding the minimum mean loss for training and testing sets
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
        "min_mean_loss_test": min_mean_loss_test,    # Add minimum mean loss for testing
    }

    # Check if model_weights/ subdir exists
    if not Path("model_weights/").exists():
        os.mkdir(Path("model_weights/"))

    # Construct the filename using current date and time
    filename = f"model_weights/model_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.pt"  # Use .pt for PyTorch models

    # Save the comprehensive model information
    torch.save(model_info, filename)

    # Create plot of losses
    figure(figsize=(9, 10), dpi=80)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    ax1.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_train], label="Train", color="blue")
    ax2.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_test], label="Test", color="red")
    fig.legend()

    # Check if /artifacts/ subdir exists
    if not Path("artifacts/").exists():
        os.mkdir(Path("artifacts/"))

    # save plot of losses
    fig.savefig(Path("artifacts") / f"session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")

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
        "--nb_epochs", help="number of training iterations", default=12, type=int
    )
    parser.add_argument("--batch_size", help="batch_size", default=4, type=int)
    parser.add_argument(
        "--balanced_batches",
        help="whether to balance batches for class labels",
        action='store_true'
    )
    parser.set_defaults(balanced_batches=False)
    args = parser.parse_args()

    main(args)
