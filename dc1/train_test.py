from torch import Tensor
from tqdm import tqdm
import torch
from dc1.net import Net
from dc1.batch_sampler import BatchSampler
from typing import Callable, List, Tuple, Any, Dict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import roc_curve, auc


def calculate_accuracy(predictions, labels):
    _, predicted = torch.max(predictions.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct, total


def transform_labels(labels):
    # 'No Finding' is class 3, considered negative (0); all other classes are positive (1)
    return [0 if label == 3 else 1 for label in labels]


# Label information mapping
label_names = {
    5: 'Pneumothorax',
    4: 'Nodule',
    3: 'No Finding',
    2: 'Infiltration',
    1: 'Effusion',
    0: 'Atelectasis'
}


def print_confusion_matrix(cm, label):
    df_cm = pd.DataFrame(cm, index=["Actual Positive", "Actual Negative"],
                         columns=["Predicted Positive", "Predicted Negative"])
    print(f"Confusion Matrix for {label} as Positive:")
    print(tabulate(df_cm, headers='keys', tablefmt='psql'))
    print('\n')


def train_model(
        model: Net,
        train_sampler: BatchSampler,
        optimizer: torch.optim.Optimizer,
        loss_function: Callable[..., torch.Tensor],
        device: str,
) -> tuple[list[Tensor], float, Any, Any]:
    # Lets keep track of all the losses:
    losses = []
    all_predictions = []
    all_labels = []

    # Put the model in train mode:
    model.train()

    # Feed all the batches one by one:
    for batch in tqdm(train_sampler):
        # Get a batch:
        x, y = batch
        # Making sure our samples are stored on the same device as our model:
        x, y = x.to(device), y.to(device)
        # Get predictions:
        predictions = model.forward(x)
        loss = loss_function(predictions, y)
        losses.append(loss)

        _, predicted = torch.max(predictions.data, 1)  # Get predictions
        all_predictions.extend(predicted.cpu().numpy())  # Store predictions
        all_labels.extend(y.cpu().numpy())  # Store true labels

        # We first need to make sure we reset our optimizer at the start.
        # We want to learn from each batch seperately,
        # not from the entire dataset at once.
        optimizer.zero_grad()
        # We now backpropagate our loss through our model:
        loss.backward()
        # We then make the optimizer take a step in the right direction.
        optimizer.step()

    binary_all_labels = transform_labels(all_labels)
    binary_all_predictions = transform_labels(all_predictions)

    binary_accuracy = accuracy_score(binary_all_labels, binary_all_predictions)
    binary_precision = precision_score(binary_all_labels, binary_all_predictions, zero_division=0)
    binary_recall = recall_score(binary_all_labels, binary_all_predictions, zero_division=0)

    return losses, binary_accuracy, binary_precision, binary_recall


def test_model(
        model: Net,
        test_sampler: BatchSampler,
        loss_function: Callable[..., torch.Tensor],
        device: str,
) -> tuple[list[Tensor], float, Any, Any, dict[str, float]]:
    # Setting the model to evaluation mode:
    model.eval()
    losses = []
    all_predictions = []
    all_labels = []
    all_probabilities = []

    # We need to make sure we do not update our model based on the test data:
    with torch.no_grad():
        for (x, y) in tqdm(test_sampler):
            # Making sure our samples are stored on the same device as our model:
            x = x.to(device)
            y = y.to(device)
            prediction = model.forward(x)
            loss = loss_function(prediction, y)
            losses.append(loss)

            _, predicted = torch.max(prediction.data, 1)  # Get predictions
            probabilities = torch.nn.functional.softmax(prediction, dim=1)
            all_probabilities.extend(probabilities.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())  # Store predictions
            all_labels.extend(y.cpu().numpy())  # Store true labels

    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    roc_auc_dict = {}
    num_classes = 6
    for i in range(num_classes):
        # Compute ROC curve and ROC area for each class
        fpr, tpr, _ = roc_curve(all_labels == i, all_probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        roc_auc_dict[label_names[i]] = roc_auc

    binary_all_labels = transform_labels(all_labels)
    binary_all_predictions = transform_labels(all_predictions)

    binary_accuracy = accuracy_score(binary_all_labels, binary_all_predictions)
    binary_precision = precision_score(binary_all_labels, binary_all_predictions, zero_division=0)
    binary_recall = recall_score(binary_all_labels, binary_all_predictions, zero_division=0)

    return losses, binary_accuracy, binary_precision, binary_recall, roc_auc_dict
