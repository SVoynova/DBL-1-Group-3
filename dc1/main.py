from dc1 import calibrate_model, train_test
from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from dc1.net import Net
from dc1.calibrate_model import calibrate_evaluate

from torchsummary import summary  # type: ignore
import torch
import torch.nn as nn
import numpy as np

# Other imports
import matplotlib
import matplotlib.pyplot as plt  # type: ignore
import argparse
import plotext  # type: ignore
from pathlib import Path
import os
from netcal.presentation import ReliabilityDiagram

# Utility class for the evaluation metric things
from MCDropout import MCDropoutAnalysis
from dc1.softmaxOutputDemo import print_images_with_probabilities
from evaluationMetricUtility import EvaluationMetricsLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import random
matplotlib.use('TkAgg')


def main(args: argparse.Namespace, activeloop: bool = True) -> None:
    seed_value = 15
    random.seed(seed_value)
    np.random.seed(seed_value)  # Numpy module
    torch.manual_seed(seed_value)  # PyTorch for CPU operations

    # Load the train and test datasets. Enable data augmentation by setting the augmentation flag to True.
    # The second 'False' refers to the augmentation flag (set to True to enable augmentation).
    # The first 'False' addresses class imbalance; it should be set to True for training data to handle imbalance,
    # and False for test data. If no 'labels_for_augmentation' is specified but augmentation is enabled,
    # apply augmentation to all classes.

    #
    # train_dataset = ImageDataset(Path("../data/X_train.npy"),
    #                              Path("../data/Y_train.npy"), False, True, [0, 1, 2, 3, 4, 5])
    # test_dataset = ImageDataset(Path("../data/X_test.npy"),
    #                             Path("../data/Y_test.npy"), False, False, [0, 1, 2, 3, 4, 5])
    #
    # #
    adjust_class_weights = False  # Set this to true to enable class weights adjustment using the validation set

    # For training dataset
    train_dataset = ImageDataset(Path("../data/X_train.npy"), Path("../data/Y_train.npy"), False, False, False)

    # For validation dataset
    validation_dataset = ImageDataset(Path("../data/X_train.npy"), Path("../data/Y_train.npy"), False, False, is_validation=True, split_ratio=0.1)

#    validation_dataset = ImageDataset(Path("../data/X_train.npy"), Path("../data/Y_train.npy"), True, False,
#                                      is_validation=True, split_ratio=0.1)
    # Test dataset remains the same
    test_dataset = ImageDataset(Path("../data/X_test.npy"), Path("../data/Y_test.npy"), False, False, False)

    MonteCarlo = False
    Calibration = True

    # Initialize the Neural Net with the number of distinct labels
    #Depth determins the number of layers employed, MCd to True for Monte Carlo Dropout
    model = Net(n_classes=6,  depth = 3, MCd = MonteCarlo)
    #LOAD WEIGHTS OF SAVED MODEL

    #model.load_state_dict(torch.load("../dc1/V.pth"))

    optimizer = AdamW(model.parameters(), lr=0.00025)  # AdamW requires a lower LR generally

    # Define a scheduler as before
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    #Modified loss function to compensate for class imbalances
    #class_weights = torch.tensor([2, 2, 2, 1.0, 3, 4], dtype=torch.float)
    #class_weights = torch.tensor([2., 2.5, 2, 1.4, 4, 4.8], dtype=torch.float)

    class_weights = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.float)

    # Modified loss function to compensate for class imbalances
    # class_weights = torch.tensor([0.8, 2.06, 2.42, 2.63, 3.74, 4.69], dtype=torch.float)
    # class_weights = torch.tensor([2, 2, 2, 1.0, 3, 4], dtype=torch.float)
    #class_weights = torch.tensor([2., 2, 2, 1, 3, 4], dtype=torch.float)
    # class_weights = torch.tensor([1., 1., 1, 1.0, 1, 1], dtype=torch.float)
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

    validation_sampler = BatchSampler(
         batch_size=args.batch_size, dataset=validation_dataset, balanced=args.balanced_batches
         # Assuming balanced=False disables balancing
    )

    test_sampler = BatchSampler(
        batch_size=100, dataset=test_dataset, balanced=args.balanced_batches
    )

    # Instantiate EvaluationMetricsLogger for logging and plotting metrics
    metrics_logger = EvaluationMetricsLogger()

    if torch.cuda.is_available():
        class_weights = class_weights.to('cuda')
    elif torch.backends.mps.is_available():
        class_weights = class_weights.to('mps')  # Ensure class weights are also moved to MPS
    else:
        class_weights = class_weights.to('cpu')
    loss_function = nn.CrossEntropyLoss(weight=class_weights)


    softmaxList = []

    # #Variables for going back to better model
    # lowest_test_error = float('inf')
    # epochs_since_improvement = 0
    # rollback_threshold = 2  # Number of epochs to wait before rolling back
    # best_model_state = None  # This will hold the best model's state in memory

    min_loss = 1000

    # Loop over epochs to train and test the model, logging the metrics
    for e in range(n_epochs):
        if activeloop:
            metrics_logger.log_training_epochs(e, model, train_sampler, optimizer, loss_function, device, MCd=False)

            if adjust_class_weights:
                # Update class weights based on validation
                class_weights = metrics_logger.validate_and_adjust_weights(e, model, validation_sampler, device,
                                                                           class_weights, adjustment_factor=1.0)
                # Update the loss function with new class weights
                loss_function = nn.CrossEntropyLoss(weight=class_weights)

            training_loss = metrics_logger.log_testing_epochs(e, model, test_sampler, loss_function, device)

            if training_loss.item() < min_loss:
                torch.save(model.state_dict(), 'BEST_WEIGHTS.pth')
                min_loss = training_loss

            if MonteCarlo:
                avg = metrics_logger.log_testing_epochs(e, model, test_sampler, loss_function, device)
                softmaxList.append(avg)

            # Plots the both training and the testing losses of the logged model
            metrics_logger.plot_training_testing_losses()

            scheduler.step()

    if not Path("model_weights/").exists():
        os.mkdir(Path("model_weights/"))

    if MonteCarlo:
        analysis = MCDropoutAnalysis(softmaxList)
        # To print statistics
        analysis.print_statistics()
        # To plot the probability distributions
        analysis.plot_probability_distributions()
        # To plot the trend analysis
        analysis.plot_trend_analysis()

    # Saves the model
    metrics_logger.save_model(model, n_epochs, batch_size)

    # Plot overall loss and accuracy, and ROC curve for model evaluation
    # RE-ADD AFTER
    metrics_logger.plot_loss_and_accuracy(n_epochs)
    metrics_logger.plot_roc_curve(n_epochs, train_test)

    #metrics_logger.plot_loss_and_accuracy(n_epochs)
    #metrics_logger.plot_roc_curve(n_epochs, train_test)

    model.eval()

    # SoftMax Demo before calibration
    #print_images_with_probabilities(model, test_dataset, device)

    # Even if no calibration is implemented, this is useful
    # Calculate ECE , Brier score and reliability diagram before calibration
    metrics_logger.calculate_and_log_ece()
    # Plot reliability diagram for each class before calibration, saved in directory graphs
    for i in range(0, 6):
        metrics_logger.multiclass_calibration_curve(i, False)

    if Calibration:
        # Calibrate model using optimal temperature scaling and evaluate
        # This calculates ECE, Brier score and plots reliability diagrams after calibration
        opt_temperature = calibrate_model.calibrate_evaluate(model, validation_sampler, test_dataset, loss_function,
                                                             device)

    # SoftMax Demo
    #print_images_with_probabilities(model, test_dataset, device)

    #SAVES MODEL WEIGHTS OF FINAL EPOCH
    torch.save(model.state_dict(), 'FINAL_WEIGHTS.pth')

def setup_model_device(model, DEBUG):
    """
    Move the model to the appropriate device based on availability and debug flag.
    """
    # Moving our model to the right device (CUDA will speed training up significantly!)
    if torch.cuda.is_available() and not DEBUG:
        #torch.cuda.manual_seed(seed_value)
        print("@@@ CUDA device found, enabling CUDA training...")
        device = "cuda"
        model.to(device)
        # Creating a summary of our model and its layers:
        summary(model, (1, 128, 128), device=device)
    elif torch.backends.mps.is_available() and not DEBUG:  # PyTorch supports Apple Silicon GPU's from version 1.12
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
    parser.add_argument("--batch_size", help="batch_size", default=50, type=int)
    parser.add_argument(
        "--balanced_batches",
        help="whether to balance batches for class labels",
        action='store_true'
    )
    parser.set_defaults(balanced_batches=True)
    args = parser.parse_args()

    main(args)