from pathlib import Path
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np

def extract_and_structure_model_info(file_path):
    """
    Load model information from a file and structure it into a dictionary.
    """
    loaded_contents = torch.load(file_path)

    structured_info = {
        "layers": {},
        "n_epochs": loaded_contents.get("n_epochs", None),
        "batch_size": loaded_contents.get("batch_size", None),
        "mean_losses_train": loaded_contents.get("mean_losses_train", []),
        "mean_losses_test": loaded_contents.get("mean_losses_test", []),
        "min_mean_loss_train": loaded_contents.get("min_mean_loss_train", None),  # New info
        "min_mean_loss_test": loaded_contents.get("min_mean_loss_test", None)  # New info
    }

    state_dict = loaded_contents.get("state_dict", {})
    for layer_name, params in state_dict.items():
        structured_info["layers"][layer_name] = {
            "parameter_count": params.numel(),
            "size": list(params.size())
        }

    return structured_info

def process_all_models(directory):
    """
    Process all model files in a given directory and return their structured information.
    Only processes files with a .pt extension.

    Parameters:
    - directory (str): Path to the directory containing model files.

    Returns:
    - list: A list of dictionaries with structured information for each model.
    """
    all_models_info = []
    base_dir = Path(directory)

    # Loop over all .pt files in the directory
    # The glob pattern "*.pt" matches only files that end with '.pt'
    for file_path in base_dir.glob("*.pt"):
        model_info = extract_and_structure_model_info(file_path)
        all_models_info.append(model_info)

    return all_models_info

def display_model_info(all_models_info, filename="model_info.txt"):
    """
    Display structured information for each model in a readable format and save it to a file.
    """
    with open(filename, "w") as file:
        for model_idx, model_info in enumerate(all_models_info, start=1):
            model_output = f"Model {model_idx}:\n"
            model_output += f"  Number of Epochs: {model_info['n_epochs']}\n"
            model_output += f"  Batch Size: {model_info['batch_size']}\n"
            model_output += f"  Mean Losses Train: {model_info['mean_losses_train']}\n"
            model_output += f"  Mean Losses Test: {model_info['mean_losses_test']}\n"
            model_output += f"  Minimum Mean Losses Train: {model_info['min_mean_loss_train']}\n"  # New info
            model_output += f"  Minimum Mean Losses Test: {model_info['min_mean_loss_test']}\n"  # New info
            model_output += "  Layers:\n"
            for layer_name, layer_info in model_info["layers"].items():
                model_output += f"    Layer: {layer_name}\n"
                model_output += f"      Parameter Count: {layer_info['parameter_count']}\n"
                model_output += f"      Size: {layer_info['size']}\n"
            model_output += "\n"

            # Print to console
            print(model_output)

            # Write to file
            file.write(model_output)


def plot_error(all_models_info):
    """
    Plot both the training and testing error for each different number of iterations,
    with each plot representing a unique iteration count. Colors are determined by batch size,
    with solid lines for training data and dashed lines for testing data.
    """
    # Extract unique batch sizes and map them to colors
    unique_batch_sizes = sorted(set(info['batch_size'] for info in all_models_info if info['batch_size'] is not None))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_batch_sizes)))  # Generate a colormap

    batch_size_to_color = {batch_size: color for batch_size, color in zip(unique_batch_sizes, colors)}

    for iteration_count in sorted(set(info['n_epochs'] for info in all_models_info if info['n_epochs'] is not None)):
        models_with_iteration = [info for info in all_models_info if info['n_epochs'] == iteration_count]

        plt.figure(figsize=(10, 6))

        for model_info in models_with_iteration:
            train_errors = model_info['mean_losses_train']
            test_errors = model_info['mean_losses_test']
            batch_size = model_info['batch_size']
            color = batch_size_to_color[batch_size]

            # Plot settings
            epochs_train = list(range(1, len(train_errors) + 1))
            epochs_test = list(range(1, len(test_errors) + 1))

            plt.plot(epochs_train, train_errors, color=color, linestyle='-', label=f'Batch Size: {batch_size} Training')
            plt.plot(epochs_test, test_errors, color=color, linestyle='--', label=f'Batch Size: {batch_size} Testing')

        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title(f'Error by Epoch for {iteration_count} Iterations')
        # To reduce legend clutter, you might consider creating a custom legend
        # that summarizes the color-to-batch-size mapping and differentiates between line styles.
        plt.legend()
        plt.show()


directory = "model_weights"  # Adjust the path as necessary
all_models_info = process_all_models(directory)
#display_model_info(all_models_info)
plot_error(all_models_info)

