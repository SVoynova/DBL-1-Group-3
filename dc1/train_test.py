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
import matplotlib.pyplot as plt

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
activations = {}

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

    datadis = [0] * 6
    labels_distr = [[0] * 6 for _ in range(6)]  # stores every prediction made, arranged by class
    for i in range(len(all_predictions)):
        if all_predictions[i] == all_labels[i]:
            labels_distr[all_labels[i]][all_labels[i]] += 1
        else:
            labels_distr[all_labels[i]][all_predictions[i]] += 1
        datadis[all_labels[i]] += 1

    binary_all_labels = transform_labels(all_labels)
    binary_all_predictions = transform_labels(all_predictions)

    binary_accuracy = accuracy_score(binary_all_labels, binary_all_predictions)
    binary_precision = precision_score(binary_all_labels, binary_all_predictions, zero_division=0)
    binary_recall = recall_score(binary_all_labels, binary_all_predictions, zero_division=0)

    return losses, binary_accuracy, binary_precision, binary_recall, labels_distr, datadis


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
    last_image = None

    #Softmax output store for MC dropout
    softmaxLis = []

    # Register hook for the layers you're interested in
    printHeatmap = False
    if printHeatmap:
        model.layer1[0].register_forward_hook(get_activation('layer1_0'))
        model.layer2[0].register_forward_hook(get_activation('layer2_0'))


    # We need to make sure we do not update our model based on the test data:
    with torch.no_grad():
        for (x, y) in tqdm(test_sampler):
            activations.clear()
            # Making sure our samples are stored on the same device as our model:
            x = x.to(device)
            y = y.to(device)
            prediction = model.forward(x)

            #For MC-Dropout, storing average of Softmax per batch
            average_prediction = torch.mean(prediction, dim=0)
            softmaxLis.append(average_prediction)


            #            prediction = model(x)  # Forward pass
            loss = loss_function(prediction, y)
            losses.append(loss)

            _, predicted = torch.max(prediction.data, 1)  # Get predictions
            probabilities = torch.nn.functional.softmax(prediction, dim=1)
            all_probabilities.extend(probabilities.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())  # Store predictions
            all_labels.extend(y.cpu().numpy())  # Store true labels


            last_image = x[-1]  # Save the last image processed

    if printHeatmap:
        last_image = None
        if last_image is not None:
            # Let's visualize the first feature map of layer1_0 activations
            feature_map = activations['layer1_0'][0][0].cpu().numpy()  # Indexing: [batch, feature_map, :, :]
            plt.imshow(feature_map, cmap='hot')
            plt.colorbar()
            plt.show()

            feature_map = activations['layer2_0'][0][0].cpu().numpy()  # Indexing: [batch, feature_map, :, :]
            plt.imshow(feature_map, cmap='hot')
            plt.colorbar()
            plt.show()

    datadis = [0] * 6
    labels_distr = [[0] * 6 for _ in range(6)]  # stores every prediction made, arranged by class
    for i in range(len(all_predictions)):
        if all_predictions[i] == all_labels[i]:
            labels_distr[all_labels[i]][all_labels[i]] += 1
        else:
            labels_distr[all_labels[i]][all_predictions[i]] += 1
        datadis[all_labels[i]] += 1


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

    #checking if last image is even selected
    # if last_image is not None:
    #     plt.imshow(last_image.squeeze(), cmap='gray')  # Assuming the image is grayscale
    #     plt.title("Last Test Image")
    #     plt.colorbar()
    #     plt.show()


#    target_layer = model.cnn_layers[-2]  # Adjust based on which layer you're interested in
#    grad_cam = GradCAM(model, target_layer)

    # Visualize the heatmap for your input image
    # This will generate a heatmap visualization for the top predicted class
    # You can specify a target_class as an integer if you want to visualize for a specific class
 #   grad_cam.visualize(last_image, target_class=None)

    # Test loop - unchanged
    # with torch.no_grad():
    #     for (x, y) in tqdm(test_sampler):
    #         # Existing processing...
    #         last_image = x[-1].unsqueeze(0)  # Save the last image of the batch
    #
    # # Now generate the heatmap for the last image processed
    # if last_image is not None:
    #     last_image = last_image.to(device)
    #     heatmap, _ = model.get_heatmap(last_image)  # Adjusted to receive heatmap directly
    #     # Assuming the original last image is single-channel (grayscale), we can visualize it directly
    #     plt.imshow(last_image.cpu().squeeze(), cmap='gray')
    #     plt.imshow(heatmap.cpu().squeeze(), cmap='jet', alpha=0.5)  # Overlay heatmap
    #     plt.colorbar()
    #     plt.show()

    #For MC-Dropout, takes average over entire batch
    softmax_tensor = torch.stack(softmaxLis)
    overall_average = torch.mean(softmax_tensor, dim=0)

    #return losses, binary_accuracy, binary_precision, binary_recall, roc_auc_dict, labels_distr,datadis,overall_average
    #return losses, binary_accuracy, binary_precision, binary_recall, roc_auc_dict, labels_distr, datadis, all_probabilities, all_labels
    #return losses, binary_accuracy, binary_precision, binary_recall, roc_auc_dict, labels_distr, datadis

    return losses, binary_accuracy, binary_precision, binary_recall, roc_auc_dict, labels_distr,datadis, overall_average, all_probabilities, all_labels

def get_activation(name):
    # This function will return another function that will be used as a hook
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def visualize_activations(activations):
    for name, activation in activations.items():
        # Assuming we're visualizing the first activation map of each layer
        act = activation[0][0].cpu().numpy()  # Getting the first feature map
        plt.figure(figsize=(10, 5))
        plt.title(f"Activation: {name}")
        plt.imshow(act, cmap='hot')
        plt.colorbar()
        plt.show()

