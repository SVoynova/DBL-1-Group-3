import torch
from torch.utils.data import DataLoader
import numpy as np
from temperature_scaling import TemperatureScaling
from sklearn.metrics import brier_score_loss
from netcal.metrics import ECE
from netcal.presentation import ReliabilityDiagram
import matplotlib.pyplot as plt



def calibrate_evaluate(model, validation_sampler, test_dataset, device):

    model.eval()

    # Initialize the temperature scaling model
    temp_scaling_model = TemperatureScaling().to(device)

    # Calibrate the model
    temp_scaling_model.set_temperature(validation_sampler, model, device)
    optimal_temperatue = temp_scaling_model.temperature.item()
    print("Temperature:", optimal_temperatue)

    #post_calibration_metrics = EvaluationMetricsLogger()

    # Assuming test_dataset is already defined as shown in your snippet
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Prepare your data loader
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    predictions, targets = [], []

    # Collect predictions and targets
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            scaled_logits = logits / optimal_temperatue
            probs = torch.nn.functional.softmax(scaled_logits, dim=1)
            predictions.extend(probs.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.eye(6)[targets]  # Convert to one-hot encoding for Brier score

    # Calculate metrics
    ece = ECE(15)  # Using 15 bins for ECE
    ece_value = ece.measure(predictions, targets.argmax(axis=1))
    average_brier_score = multi_class_brier_score(targets, predictions)

    #brier_score = brier_score_loss(targets, predictions, pos_label=1)

    print(f"Expected Calibration Error (ECE) after calibration: {ece_value}")
    print("Average Brier Score for Multi-Class:", average_brier_score)

    # Plot reliability diagram
    diagram = ReliabilityDiagram(15)
    diagram.plot(predictions, targets.argmax(axis=1))
    plt.show()
    
    return optimal_temperatue


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
