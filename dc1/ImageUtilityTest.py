from dc1.image_dataset import ImageDataset
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import Counter

train_dataset = ImageDataset(Path("../data/X_train.npy"),
                             Path("../data/Y_train.npy"))

label_counts = Counter(train_dataset.targets)
label_counts_dict = dict(label_counts)

max_label = max(label_counts_dict, key=label_counts_dict.get)
max_value = label_counts_dict[max_label]
print(max_label, max_value)
print("___")
for key, value in label_counts_dict.items():
    if key != max_label:
        print(key, value, int(max_value/value))

def plot_image_for_label(dataset, label_to_find):
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label == label_to_find:
            image, label = dataset[i]

            if isinstance(image, torch.Tensor):
                image = image.numpy()

            if image.shape[0] < image.shape[2]:
                image = np.transpose(image, (1, 2, 0))

            if image.shape[2] == 1:
                image = image.squeeze()

            plt.figure()
            plt.imshow(image, cmap='gray' if image.ndim == 2 else None)
            plt.title(f"Label: {label}")
            plt.axis('off')
            plt.show()

            return image
    return None

# Retrieve two images with specified labels
image1 = plot_image_for_label(train_dataset, 0)
image2 = plot_image_for_label(train_dataset, 1)

# Plot the overlapped images
plt.figure()
plt.imshow(image1, cmap='gray' if image1.ndim == 2 else None)
plt.imshow(image2, cmap='gray' if image2.ndim == 2 else None, alpha=0.5)
plt.axis('off')
plt.show()
