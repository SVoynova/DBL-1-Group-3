from tqdm import tqdm
import torch
from dc1.net import Net
from dc1.batch_sampler import BatchSampler
from typing import Callable, List
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from tabulate import tabulate

def calculate_accuracy(predictions, labels):
    _, predicted = torch.max(predictions.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct, total

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
) -> List[torch.Tensor]:
    # Lets keep track of all the losses:
    losses = []

    #Add+
    correct_predictions = 0
    total_samples = 0

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

        # Add+
        _, predicted = torch.max(predictions.data, 1)
        correct_predictions += (predicted == y).sum().item()
        total_samples += y.size(0)

        # We first need to make sure we reset our optimizer at the start.
        # We want to learn from each batch seperately,
        # not from the entire dataset at once.
        optimizer.zero_grad()
        # We now backpropagate our loss through our model:
        loss.backward()
        # We then make the optimizer take a step in the right direction.
        optimizer.step()

    accuracy = 100 * correct_predictions / total_samples
    print(f'Training Accuracy: {accuracy:.2f}%')

    return losses


def test_model(
        model: Net,
        test_sampler: BatchSampler,
        loss_function: Callable[..., torch.Tensor],
        device: str,
) -> List[torch.Tensor]:
    # Setting the model to evaluation mode:
    model.eval()
    losses = []

    #Add+
    correct_predictions = 0
    total_samples = 0

    #Add+
    all_predictions = []
    all_labels = []

    # We need to make sure we do not update our model based on the test data:
    with torch.no_grad():
        for (x, y) in tqdm(test_sampler):
            # Making sure our samples are stored on the same device as our model:
            x = x.to(device)
            y = y.to(device)
            prediction = model.forward(x)
            loss = loss_function(prediction, y)
            losses.append(loss)

            #Add+
            _, predicted = torch.max(prediction.data, 1)
            correct_predictions += (predicted == y).sum().item()
            total_samples += y.size(0)

            #Add+
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    #Add+
    accuracy = 100 * correct_predictions / total_samples
    print(f'Testing Accuracy: {accuracy:.2f}%')

    #Add+
    num_classes = len(np.unique(all_labels))  # Determine the number of unique classes

    for i in range(num_classes):
        binary_labels = np.array(all_labels) == i
        binary_predictions = np.array(all_predictions) == i

        cm_binary = confusion_matrix(binary_labels, binary_predictions)
        descriptive_label = label_names.get(i, f"Class {i}")

        # Now use the print_confusion_matrix function to print each confusion matrix
        #print_confusion_matrix(cm_binary, descriptive_label)

    return losses

