# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 02:41:52 2023

@author: Shahir
"""

import os
import abc
from enum import Enum
from typing import Self, Union, Optional

import numpy as np
import numpy.typing as npt

import torch
from torch.utils.data import Dataset

from PIL.Image import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.base import BaseEstimator
from umap import UMAP

from cs168_project.datasets import DatasetType, get_dataloaders, IMAGE_TORCH_DATASETS


class AlgorithmType(str, Enum):
    PCA = "pca"
    TSNE = "tsne"
    LLE = "lle"
    UMAP = "umap"
    AE = "ae"
    AE_UMAP = "ae-umap"


ALGORITHM_TO_LABEL: dict[AlgorithmType, str] = {
    AlgorithmType.PCA: "PCA",
    AlgorithmType.TSNE: "t-SNE",
    AlgorithmType.LLE: "LLE",
    AlgorithmType.UMAP: "UMAP",
    AlgorithmType.AE: "AE",
    AlgorithmType.AE_UMAP: "AE + UMAP",
}


class BaseReducerAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
            self: Self,
            dataset_type: DatasetType,
            dataset: Dataset,
            embed_dim: int,
            seed: Optional[int] = None) -> None:

        self.dataset_type = dataset_type
        self.dataset = dataset
        self.embed_dim = embed_dim
        self.seed = seed

    @abc.abstractmethod
    def initialize(
            self: Self) -> None:

        pass

    @abc.abstractmethod
    def run(
            self: Self) -> None:

        pass

    @abc.abstractmethod
    def get_embeddings(
            self: Self) -> npt.NDArray[np.float64]:

        pass

    def save(
            self: Self,
            save_dir: os.PathLike) -> None:

        return

    def load(
            self: Self,
            save_dir: os.PathLike) -> Self:

        return


def labeled_dataset_to_x_data_np_flat(
        dataset_type: DatasetType,
        dataset: Dataset[tuple[Union[Image, npt.NDArray[np.float64]], int]]) -> npt.NDArray[np.float64]:

    if dataset_type in IMAGE_TORCH_DATASETS:
        x_data = torch.stack([x for x, y in dataset], dim=0).cpu().numpy()
    else:
        x_data = np.stack([x for x, y in dataset], axis=0)
    x_data = x_data.reshape((x_data.shape[0], -1))

    return x_data


class SKLearnLikeAlgorithm(BaseReducerAlgorithm):
    X_EMB_FNAME = "x_emb.npy"

    estimator: BaseEstimator
    x_emb: npt.NDArray[np.float64]

    def __init__(
            self: Self,
            dataset_type: DatasetType,
            dataset: Dataset,
            embed_dim: int) -> None:

        super().__init__(
            dataset_type=dataset_type,
            dataset=dataset,
            embed_dim=embed_dim)

    def run(
            self: Self) -> None:

        x_data = labeled_dataset_to_x_data_np_flat(self.dataset_type, self.dataset)
        self.x_emb = self.estimator.fit_transform(x_data)

    def get_embeddings(
            self: Self) -> npt.NDArray[np.float64]:

        return self.x_emb

    def save(
            self: Self,
            save_dir: os.PathLike) -> None:

        fname = os.path.join(save_dir, self.X_EMB_FNAME)
        np.save(fname, self.x_emb)

    def load(
            self: Self,
            save_dir: os.PathLike) -> Self:

        self.initialize()
        fname = os.path.join(save_dir, self.X_EMB_FNAME)
        self.x_emb = np.load(fname)


class PCAAlgorithm(SKLearnLikeAlgorithm):
    def __init__(
            self: Self,
            dataset_type: DatasetType,
            dataset: Dataset,
            embed_dim: int) -> None:

        super().__init__(
            dataset_type=dataset_type,
            dataset=dataset,
            embed_dim=embed_dim)

    def initialize(
            self: Self):

        self.estimator = PCA(
            n_components=self.embed_dim,
            random_state=self.seed)


class TSNEAlgorithm(SKLearnLikeAlgorithm):
    def __init__(
            self: Self,
            dataset_type: DatasetType,
            dataset: Dataset,
            embed_dim: int,
            seed: Optional[int],
            perplexity: int = 50):

        super().__init__(
            dataset_type=dataset_type,
            dataset=dataset,
            embed_dim=embed_dim)
        self.seed = seed
        self.perpexity = perplexity

    def initialize(
            self: Self):

        self.estimator = TSNE(
            n_components=self.embed_dim,
            perplexity=self.perpexity,
            random_state=self.seed)


class LLEAlgorithm(SKLearnLikeAlgorithm):
    def __init__(
            self: Self,
            dataset_type: DatasetType,
            dataset: Dataset,
            embed_dim: int,
            seed: Optional[int]):

        super().__init__(
            dataset_type=dataset_type,
            dataset=dataset,
            embed_dim=embed_dim)

        self.seed = seed
        match dataset_type:
            case DatasetType.MNIST | DatasetType.FASHION_MNIST | DatasetType.CIFAR10 | DatasetType.CIFAR100:
                self.num_neighbors = 30
            case DatasetType.SKLEARN_IRIS:
                self.num_neighbors = 10
            case DatasetType.SKLEARN_DIGITS:
                self.num_neighbors = 20
            case DatasetType.SKLEARN_CUSTOM:
                self.num_neighbors = int(len(dataset) * 0.005)
            case _:
                raise ValueError()

    def initialize(
            self: Self):

        self.estimator = LocallyLinearEmbedding(
            n_components=self.embed_dim,
            n_neighbors=self.num_neighbors,
            random_state=self.seed)


class UMAPAlgorithm(SKLearnLikeAlgorithm):
    def __init__(
            self: Self,
            dataset_type: DatasetType,
            dataset: Dataset,
            embed_dim: int,
            seed: Optional[int]):

        super().__init__(
            dataset_type=dataset_type,
            dataset=dataset,
            embed_dim=embed_dim)

        self.seed = seed
        match dataset_type:
            case DatasetType.MNIST | DatasetType.FASHION_MNIST | DatasetType.CIFAR10 | DatasetType.CIFAR100:
                self.num_neighbors = 30
            case DatasetType.SKLEARN_IRIS:
                self.num_neighbors = 10
            case DatasetType.SKLEARN_DIGITS:
                self.num_neighbors = 20
            case DatasetType.SKLEARN_CUSTOM:
                self.num_neighbors = int(len(dataset) * 0.005)
            case _:
                raise ValueError()

    def initialize(
            self: Self):

        self.estimator = UMAP(
            n_components=self.embed_dim,
            n_neighbors=self.num_neighbors,
            random_state=self.seed)


class AEAlgorithm(BaseReducerAlgorithm):
    pass


class AEUMAPAlgorithm(BaseReducerAlgorithm):
    pass
