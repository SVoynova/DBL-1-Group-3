import numpy as np
import torch
import requests
import io
from os import path
from typing import Tuple
from pathlib import Path
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math


class ImageDataset:
    """
    Creates a DataSet from numpy arrays while keeping the data
    in the more efficient numpy arrays for as long as possible and only
    converting to torch tensors when needed (torch tensors are the objects used
    to pass the data through the neural network and apply weights).
    If the flag {@code augment} is set to True, it applies data augmentation techniques to
    the images upon retrieval, based on the specified labels in {@code labels_for_augmentation}.
    The default is False, meaning no augmentation is applied.

    @params:
    - x (Path): path to the numpy file containing the images.
    - y (Path): path to the numpy file containing the labels.
    - augment (bool, optional): If True, applies data augmentation techniques to
      the images upon retrieval. Default is False, meaning no augmentation is applied.
    - labels_for_augmentation (List[int], optional): Specifies the labels of images
      to which the augmentation should be applied. If None, augmentation is applied
      to no images. Only effective if {@code augment} is True.
    """

    def __init__(self, x: Path, y: Path, balance_dataset, augment, is_validation=False, split_ratio=0.2,
                 labels_for_augmentation=None) -> None:
        self.targets = np.load(y)
        self.imgs = np.load(x)

        if is_validation:
            self.imgs, self.targets = self.split_validation_set(self, split_ratio = split_ratio)

        # Augmentation flag (set in the constructor)
        self.augment = augment

        self.labels_for_augmentation = labels_for_augmentation if labels_for_augmentation is not None else []

        if self.augment and self.labels_for_augmentation is None:
            self.labels_for_augmentation = [0, 1, 2, 3, 4, 5]

        if balance_dataset:
            # Augmentation is true, and augment all the labels
            self.augment = True
            self.labels_for_augmentation = [0, 1, 2, 4, 5, 6]

            new_img = []
            new_label = []

            unique_labels, counts = np.unique(self.targets, return_counts=True)
            label_counts = dict(zip(unique_labels, counts))

            target_label_count = label_counts.get(3, 0)
            for label in label_counts:
                if label != 3:
                    ratio = (target_label_count / label_counts[label]) * 0.1
                    label_counts[label] = math.floor(ratio) - (1 if label in [0, 2] else 0)
            a = [[label, count] for label, count in label_counts.items()]

            # print(a)

            add = 0
            for k in range(len(self.targets)):
                new_img.append(self.imgs[k])
                new_label.append(self.targets[k])
                if self.targets[k] != 3:
                    for i in range(a[self.targets[k]][1]):
                        new_img.insert(k + add, self.imgs[k])
                        new_label.insert(k + add, self.targets[k])
                        add += 1

            self.targets = np.array(new_label)
            self.imgs = np.array(new_img)

        if self.augment:
            # Defined an ImageDataGenerator including centering, normalization, randomly fliping h/v, shifts h,w and rotation.
            self.datagen = ImageDataGenerator(
                samplewise_center=True,
                samplewise_std_normalization=True,
                horizontal_flip=True,
                vertical_flip=False,
                height_shift_range=0.01,
                width_shift_range=0.02,
                rotation_range=2,
                shear_range=0.01,
                fill_mode='reflect',
                zoom_range=0.05)

            # Fit the data generator to the images for augmentation
            self.datagen.fit(self.imgs)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        image = self.imgs[idx]
        label = self.targets[idx]

        if self.augment and label in self.labels_for_augmentation:
            # Only apply augmentation if the image's label is in the specified labels for augmentation
            image = np.transpose(image, (1, 2, 0))
            image = image.reshape((1,) + image.shape)
            image = next(self.datagen.flow(image, batch_size=1))[0]  # Augmentate
            image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image / 255).float()  # Normalize and convert to tensor
        return image, label

    def split_validation_set(self, split_ratio):

        total_samples = len(self.targets)
        num_val_samples = int(total_samples * split_ratio)

        # shuffle before splitting
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        val_indices = indices[:num_val_samples]
        train_indices = indices[num_val_samples:]

        # split the data
        val_imgs = self.imgs[val_indices]
        val_targets = self.targets[val_indices]
        self.imgs = self.imgs[train_indices]
        self.targets = self.targets[train_indices]

        return val_imgs, val_targets

    @staticmethod
    def load_numpy_arr_from_npy(path: Path) -> np.ndarray:
        """
        Loads a numpy array from local storage.

        Input:
        path: local path of file

        Outputs:
        dataset: numpy array with input features or labels
        """

        return np.load(path)


def load_numpy_arr_from_url(url: str) -> np.ndarray:
    """
    Loads a numpy array from surfdrive.

    Input:
    url: Download link of dataset

    Outputs:
    dataset: numpy array with input features or labels
    """

    response = requests.get(url)
    response.raise_for_status()

    return np.load(io.BytesIO(response.content))


if __name__ == "__main__":
    cwd = os.getcwd()
    if path.exists(path.join(cwd + "data/")):
        print("Data directory exists, files may be overwritten!")
    else:
        os.mkdir(path.join(cwd, "../data/"))

    ### Load labels
    train_y = load_numpy_arr_from_url(
        url="https://surfdrive.surf.nl/files/index.php/s/i6MvQ8nqoiQ9Tci/download"
    )
    np.save("../data/Y_train.npy", train_y)
    test_y = load_numpy_arr_from_url(
        url="https://surfdrive.surf.nl/files/index.php/s/wLXiOjVAW4AWlXY/download"
    )
    np.save("../data/Y_test.npy", test_y)
    ### Load data
    train_x = load_numpy_arr_from_url(
        url="https://surfdrive.surf.nl/files/index.php/s/4rwSf9SYO1ydGtK/download"
    )
    np.save("../data/X_train.npy", train_x)
    test_x = load_numpy_arr_from_url(
        url="https://surfdrive.surf.nl/files/index.php/s/dvY2LpvFo6dHef0/download"
    )
    np.save("../data/X_test.npy", test_x)
