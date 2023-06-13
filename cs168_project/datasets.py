# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 01:57:13 2023

@author: Shahir
"""

import os
from enum import Enum
from typing import Self, Any, Optional, TypeVar, Generic
from collections.abc import Callable, Iterable

import numpy as np
import numpy.typing as npt

from torch.utils.data import DataLoader, Dataset, ConcatDataset

import torchvision
from torchvision import transforms

from PIL.Image import Image
from sklearn.datasets import make_classification, load_iris, load_digits

from cs168_project.utils import get_rel_pkg_path


DEFAULT_DATA_DIR = get_rel_pkg_path("resources/data/")
os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)


class DatasetType(str, Enum):
    SKLEARN_CUSTOM = "sklearn-custom"
    SKLEARN_IRIS = "sklearn-iris"
    SKLEARN_DIGITS = "sklearn-digits"
    MNIST = "mnist"
    FASHION_MNIST = "fashion-mnist"
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"


DATASET_TO_LABEL: dict[DatasetType, str] = {
    DatasetType.SKLEARN_CUSTOM: "Scikit-learn Custom",
    DatasetType.SKLEARN_IRIS: "Iris flowers",
    DatasetType.SKLEARN_DIGITS: "UCI ML digits",
    DatasetType.MNIST: "MNIST",
    DatasetType.FASHION_MNIST: "Fashion MNIST",
    DatasetType.CIFAR10: "CIFAR-10",
    DatasetType.CIFAR100: "CIFAR-1000"
}
IMAGE_TORCH_DATASETS: list[DatasetType] = [
    DatasetType.MNIST,
    DatasetType.FASHION_MNIST,
    DatasetType.CIFAR10,
    DatasetType.CIFAR100
]
FLAT_NP_DATASETS: list[DatasetType] = [
    DatasetType.SKLEARN_CUSTOM,
    DatasetType.SKLEARN_IRIS,
    DatasetType.SKLEARN_DIGITS
]
DATASET_TO_SHAPE: dict[DatasetType, Iterable[int]] = {
    DatasetType.MNIST: (1, 28, 28),
    DatasetType.FASHION_MNIST: (1, 28, 28),
    DatasetType.CIFAR10: (3, 32, 32),
    DatasetType.CIFAR100: (3, 32, 32),
    DatasetType.SKLEARN_IRIS: (4,),
    DatasetType.SKLEARN_DIGITS: (1, 8, 8)
}
DATASET_TO_NUM_SAMPLES: dict[DatasetType, int] = {
    DatasetType.MNIST: 70000,
    DatasetType.FASHION_MNIST: 70000,
    DatasetType.CIFAR10: 60000,
    DatasetType.CIFAR100: 60000,
    DatasetType.SKLEARN_IRIS: 150,
    DatasetType.SKLEARN_DIGITS: 1797
}

T_co = TypeVar('T_co', covariant=True)
T_co1 = TypeVar('T_co1', covariant=True)
T_co2 = TypeVar('T_co2', covariant=True)
T_co3 = TypeVar('T_co3', covariant=True)


def combine_columns_dataset(
        *columns: Iterable[Any]) -> list[tuple[Any]]:

    return list(zip(*columns))


class RawDataset(Generic[T_co], Dataset[T_co]):
    def __init__(
            self: Self,
            x_data: Iterable[T_co1],
            metadata: Optional[Any] = None) -> None:

        super().__init__()

        self.x_data = x_data
        self.metadata = metadata

    def __getitem__(
            self,
            index: int) -> T_co:

        return self.x_data[index]

    def __len__(
            self) -> int:

        return len(self.x_data)


def transform_with_label(
        transform_func: Callable[[T_co1], T_co2]) -> Callable[[tuple[T_co1, T_co3]], tuple[T_co2, T_co3]]:

    def transform_with_label_wrapper(
            data: tuple[T_co1, T_co3]) -> tuple[T_co2, T_co3]:

        inputs, label = data

        return transform_func(inputs), label

    return transform_with_label_wrapper


class TransformDataset(Generic[T_co1, T_co2], Dataset[T_co2]):
    def __init__(
            self: Self,
            dataset: Dataset,
            transform_func: Callable[[T_co1], T_co2]) -> None:

        super().__init__()

        self.dataset = dataset
        self.transform_func = transform_func

    def __getitem__(
            self: Self,
            index: int) -> T_co2:

        inputs = self.transform_func(self.dataset[index])

        return inputs

    def __len__(
            self) -> int:

        return len(self.dataset)


def get_img_dataset_raw(
        data_dir: str,
        dataset_type: DatasetType,
        combine_train_test: bool = False) -> dict[str, Dataset[tuple[Image, int]]]:

    match dataset_type:
        case DatasetType.MNIST:
            train_dataset = torchvision.datasets.MNIST(
                data_dir,
                train=True,
                download=True)
            test_dataset = torchvision.datasets.MNIST(
                data_dir,
                train=False,
                download=True)
        case DatasetType.FASHION_MNIST:
            train_dataset = torchvision.datasets.FashionMNIST(
                data_dir,
                train=True,
                download=True)
            test_dataset = torchvision.datasets.FashionMNIST(
                data_dir,
                train=False,
                download=True)
        case DatasetType.CIFAR10:
            train_dataset = torchvision.datasets.CIFAR10(
                data_dir,
                train=True,
                download=True)
            test_dataset = torchvision.datasets.CIFAR10(
                data_dir,
                train=False,
                download=True)
        case DatasetType.CIFAR100:
            train_dataset = torchvision.datasets.CIFAR100(
                data_dir,
                train=True,
                download=True)
            test_dataset = torchvision.datasets.CIFAR100(
                data_dir,
                train=False,
                download=True)
        case _:
            raise ValueError()

    if combine_train_test:
        out = {
            'all': ConcatDataset((train_dataset, test_dataset))
        }

        return out
    else:
        out = {
            'train': train_dataset,
            'test': test_dataset
        }

        return out


def get_img_dataset_transforms(
        dataset_type: DatasetType,
        augment: bool = True,
        new_input_size: Optional[int] = None) -> dict[str, Callable]:

    normalize_mean = None
    normalize_std = None

    match dataset_type:
        case DatasetType.MNIST | DatasetType.FASHION_MNIST:
            normalize_mean = np.array([0.1307], dtype=np.float32)
            normalize_std = np.array([0.3081], dtype=np.float32)
            to_tensor_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.1307,),
                    std=(0.3081,))])

            train_transform = transforms.Compose([
                transforms.RandomApply([
                    transforms.RandomAffine(
                        degrees=10,
                        translate=(0.1, 0.1),
                        scale=(0.9, 1.1),
                        shear=10)],
                    p=0.9)])

            test_transform = transforms.Compose([])

        case DatasetType.CIFAR10 | DatasetType.CIFAR100:
            normalize_mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
            normalize_std = np.array([0.247, 0.243, 0.261], dtype=np.float32)
            to_tensor_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.247, 0.243, 0.261))])

            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05),
                transforms.RandomApply([
                    transforms.RandomAffine(
                        degrees=10,
                        translate=(0.1, 0.1),
                        scale=(0.9, 1.1))],
                    p=0.5)])

            test_transform = transforms.Compose([])

        case _:
            raise ValueError()

    to_tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=normalize_mean.tolist(),
            std=normalize_std.tolist())])
    from_tensor_transform = transforms.Compose([
        transforms.Normalize(
            mean=(-normalize_mean / normalize_std).tolist(),
            std=(1.0 / normalize_std).tolist()),
        transforms.ToPILImage()])

    if not augment:
        train_transform = test_transform

    if new_input_size:
        train_transform = transforms.Compose([
            transforms.Resize(new_input_size),
            train_transform])
        test_transform = transforms.Compose([
            transforms.Resize(new_input_size),
            test_transform])

    train_transform = transforms.Compose([
        train_transform,
        to_tensor_transform])
    test_transform = transforms.Compose([
        test_transform,
        to_tensor_transform])

    out = {
        'train': train_transform,
        'test': test_transform,
        'to_tensor': to_tensor_transform,
        'from_tensor': from_tensor_transform
    }

    return out


def generate_sklearn_classification_dataset(
        num_samples: int,
        num_features: int = 2,
        num_informative: int = 0,
        num_redundant: int = 0,
        num_repeated: int = 0,
        num_classes: int = 2,
        num_clusters_per_class: int = 1,
        seed: int = 0) -> Dataset[tuple[npt.NDArray[np.float64], int]]:

    samples, labels = make_classification(
        n_samples=num_samples,
        n_features=num_features,
        n_informative=num_informative,
        n_redundant=num_redundant,
        n_repeated=num_repeated,
        n_classes=num_classes,
        n_clusters_per_class=num_clusters_per_class,
        random_state=seed)
    dataset = RawDataset(combine_columns_dataset(samples, labels))

    return dataset


def get_sklearn_preset_dataset(
        dataset_type: DatasetType) -> Dataset[tuple[npt.NDArray[np.float64], int]]:

    if dataset_type == DatasetType.SKLEARN_DIGITS:
        data, labels = load_digits(return_X_y=True)
        data = data.reshape((data.shape[0], 1, 8, 8))
        dataset = RawDataset(combine_columns_dataset(data, labels))
    elif dataset_type == DatasetType.SKLEARN_IRIS:
        data, labels = load_iris(return_X_y=True)
        dataset = RawDataset(combine_columns_dataset(data, labels))

    return dataset


def get_dataloaders(
        datasets: dict[str, Dataset],
        train_batch_size: int,
        test_batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = False):

    train_dataset = datasets['train']
    train_loader = DataLoader(train_dataset,
                              batch_size=train_batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_dataset = datasets['test']
    test_loader = DataLoader(test_dataset,
                             batch_size=test_batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    test_shuffle_loader = DataLoader(test_dataset,
                                     batch_size=test_batch_size,
                                     shuffle=True,
                                     num_workers=num_workers,
                                     pin_memory=pin_memory)

    return {'train': train_loader,
            'test': test_loader,
            'test_shuffle': test_shuffle_loader}
