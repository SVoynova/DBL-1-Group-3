import torch
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
import numpy as np

from dc1.evaluationMetricUtility import EvaluationMetricsLogger
from temperature_scaling import TemperatureScaling
from sklearn.metrics import brier_score_loss
from netcal.metrics import ECE
from netcal.presentation import ReliabilityDiagram
import matplotlib.pyplot as plt



def calibrate_evaluate(model, validation_sampler, test_dataset, loss_function, device):

    model.eval()

    # Initialize the temperature scaling model
    temp_scaling_model = TemperatureScaling().to(device)

    # Calibrate the model
    temp_scaling_model.set_temperature(validation_sampler, model, device)
    optimal_temperature = temp_scaling_model.temperature.item()
    print("Temperature:", optimal_temperature)

    # Assuming test_dataset is already defined as shown in your snippet
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    predictions, targets = [], []

    metrics_logger = EvaluationMetricsLogger()

    # Collect predictions and targets
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            scaled_logits = logits / optimal_temperature
            probs = torch.nn.functional.softmax(scaled_logits, dim=1)
            predictions.extend(probs.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.eye(6)[targets]  # Convert to one-hot encoding for Brier score

    # Calculate metrics
    ece = ECE(15)  # Using 15 bins for ECE
    ece_value = ece.measure(predictions, targets.argmax(axis=1))
    average_brier_score = multi_class_brier_score(targets, predictions)

    print(f"Expected Calibration Error (ECE) after calibration: {ece_value}")
    print("Average Brier Score for Multi-Class:", average_brier_score)

    # Plot reliability diagram for each class after calibration
    metrics_logger.pred_probs_test = predictions
    metrics_logger.true_labels_test = np.argmax(targets, axis=1)
    for i in range(0,6):
        metrics_logger.multiclass_calibration_curve(i, True)

    return optimal_temperature


def multi_class_brier_score(y_true, y_prob):
    """
    Calculate the Brier score for multi-class classification.

    Parameters:
    - y_true: numpy array of shape (n_samples,), true class labels as integers
    - y_prob: numpy array of shape (n_samples, n_classes), predicted probabilities

    Returns:
    - Average Brier score across all classes.
    """


    n_classes = y_prob.shape[1]

    # Convert one-hot encoded targets back to class integers
    y_true = np.argmax(y_true, axis=1)

    # Ensure y_true is of integer type
    y_true = y_true.astype(int)

    y_true_one_hot = np.eye(n_classes)[y_true]

    scores = []
    for i in range(n_classes):
        scores.append(brier_score_loss(y_true_one_hot[:, i], y_prob[:, i]))

    return np.mean(scores)
