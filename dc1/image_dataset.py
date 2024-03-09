import numpy as np
import torch
import requests
import io
from os import path
from typing import Tuple
from pathlib import Path
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ImageDataset:
    """
    Creates a DataSet from numpy arrays while keeping the data
    in the more efficient numpy arrays for as long as possible and only
    converting to torchtensors when needed (torch tensors are the objects used
    to pass the data through the neural network and apply weights).
    If flag {@code augment} set to True,  applies data augmentation techniques to
    the images upon retrieval, default is False, meaning no augmentation is
    applied.

    @params:
    - x (Path): path to the numpy file containing the images
    - y (Path): path to the numpy file containing the labels
    - augment (bool, optional): If True, applies data augmentation techniques to
      the images upon retrieval. Default is False, meaning no augmentation is
      applied.
    """

    def __init__(self, x: Path, y: Path, augment=False) -> None:
        self.targets = np.load(y)
        self.imgs = np.load(x)

        # Augmentation flag (set in the constructor)
        self.augment = augment
        if self.augment:
            # Defined an ImageDataGenerator including centering, normalization, randomly fliping h/v, shifts h,w and rotation.
            self.datagen = ImageDataGenerator(
                samplewise_center=True,
                samplewise_std_normalization=True,
                horizontal_flip=True,
                vertical_flip=False,
                height_shift_range=0.05,
                width_shift_range=0.1,
                rotation_range=5,
                shear_range=0.1,
                fill_mode='reflect',
                zoom_range=0.15)
            # Fit the data generator to the images for augmentation
            self.datagen.fit(self.imgs)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        image = self.imgs[idx]
        label = self.targets[idx]
        if self.augment:
            image = np.transpose(image, (1, 2, 0))
            image = image.reshape((1,) + image.shape)
            image = next(self.datagen.flow(image, batch_size=1))[0]  # Augmentate
            image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image / 255).float()  # Normalize and convert to tensor
        return image, label

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
