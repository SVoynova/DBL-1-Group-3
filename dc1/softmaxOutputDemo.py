import torch
import matplotlib.pyplot as plt
import numpy as np

def print_images_with_probabilities(model, dataset, device, num_images=5):
    """
    Prints images along with a text display of the probability distribution across all classes.
    :param model: Trained neural network model.
    :param dataset: Dataset containing images and labels.
    :param device: Device to run the model on ('cuda', 'mps', or 'cpu').
    :param num_images: Number of images to display.
    """
    label_names = {
        5: 'Pneumothorax',
        4: 'Nodule',
        3: 'No Finding',
        2: 'Infiltration',
        1: 'Effusion',
        0: 'Atelectasis'
    }

    model.eval()  # Set model to evaluation mode

    # DataLoader to load the data
    loader = torch.utils.data.DataLoader(dataset, batch_size=num_images, shuffle=True)

    with torch.no_grad():  # No need to track gradients
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            probabilities = probabilities.cpu().numpy()  # Move probabilities to CPU and convert to numpy array

            # Plotting images and probability texts
            fig, axs = plt.subplots(num_images, 2, figsize=(10, 2 * num_images))
            for i in range(num_images):
                axs[i, 0].imshow(images[i].cpu().squeeze(), cmap="gray")  # Assuming images are grayscale
                real_label_index = labels[i].cpu().item()
                real_label_name = label_names[real_label_index]
                axs[i, 0].set_title(f"Real: {real_label_name}")
                axs[i, 0].axis('off')

                # Constructing text for probability distribution
                prob_text = "\n".join([f"{label}: {prob*100:.2f}%" for label, prob in zip(label_names.values(), probabilities[i])])
                axs[i, 1].text(0, 0.5, prob_text, va='center')
                axs[i, 1].axis('off')

            plt.tight_layout()
            plt.show()
            break  # Only go through one batch
