# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 01:57:54 2023

@author: Shahir
"""

import os
import json
import itertools
from dataclasses import dataclass
from typing import Self, Optional

import numpy as np
import numpy.typing as npt

from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from cs168_project.datasets import DatasetType
from cs168_project.algorithm import BaseReducerAlgorithm
from cs168_project.utils import get_rel_pkg_path

DEFAULT_OUTPUT_DIR = get_rel_pkg_path("output/")
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)


@dataclass(kw_only=True)
class ExperimentStats:
    fig_fname: os.PathLike
    knn_accs: list[tuple[int, float]]


class ExperimentStatsRunner:
    NUM_KNN_TESTS = 5
    FIG_FNAME = "embed_plot.png"
    KNN_ACCS_FNAME = "knn_accs.json"

    fig: plt.Figure
    knn_accs: list[tuple[int, float]]

    def __init__(
            self: Self,
            dataset_type: DatasetType,
            dataset: Dataset,
            algorithm: BaseReducerAlgorithm) -> None:

        self.dataset_type = dataset_type
        self.dataset = dataset
        self.algorithm = algorithm
        self.fig_saved = False

    @staticmethod
    def _plot_embedding(
            x_emb: npt.NDArray[np.float64],
            y: Optional[npt.NDArray[np.int64]] = None,
            default_color: bool = True) -> plt.Figure:

        cmap = plt.get_cmap('jet')
        fig, ax = plt.subplots()
        if y is not None:
            assert len(x_emb) == len(y)
            num_labels = max(y) + 1
            default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            if num_labels <= len(default_colors) or default_color:
                colors = default_colors
            else:
                colors = [cmap(n / num_labels) for n in range(num_labels)]

            markers = ['o', '+', 'x', 'v', '^', '*']

            prop_iter = itertools.product(markers, colors)

            for label, props in zip(range(num_labels), prop_iter):
                ix = np.where(y == label)
                ax.scatter(x_emb[ix, 0], x_emb[ix, 1],
                           c=[props[1] for _ in ix],
                           label=label,
                           marker=props[0],
                           s=0.8)
            ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        else:
            ax.scatter(x_emb[:, 0], x_emb[:, 1], s=0.8)
        fig.tight_layout()
        plt.close()

        return fig

    def _compute_knn_acc(
            self: Self,
            x_emb: npt.NDArray[np.float64],
            y: npt.NDArray[np.int64]) -> dict[int]:

        match self.dataset_type:
            case DatasetType.MNIST | DatasetType.FASHION_MNIST | DatasetType.CIFAR10 | DatasetType.CIFAR100:
                num_neighbors_space = [5, 50, 100, 300, 800]
            case DatasetType.SKLEARN_DIGITS:
                num_neighbors_space = [5, 10, 40, 60, 100]
            case DatasetType.SKLEARN_IRIS:
                num_neighbors_space = [3, 5, 10, 20, 30]
            case DatasetType.SKLEARN_CUSTOM:
                num_neighbors_space = [5, 20, 80, 200, 400]
            case _:
                raise ValueError()
        assert len(num_neighbors) == self.NUM_KNN_TESTS

        acc_data = []
        for num_neighbors in num_neighbors_space:
            classifier = KNeighborsClassifier(n_neighbors=num_neighbors)
            classifier.fit(x_emb, y)
            y_pred = classifier.predict(x_emb)
            acc = float(np.sum(y_pred == y) / len(y))
            acc_data.append((num_neighbors, acc))

        return acc_data

    def run(
            self: Self) -> None:

        x_emb = self.algorithm.get_embeddings()
        labels = np.array([y for x, y in self.dataset], dtype=np.int64)
        if self.algorithm.embed_dim == 2:
            self.fig = self._plot_embedding(x_emb, labels)
        self.knn_accs = self._compute_knn_acc(x_emb, labels)

    def save(
            self: Self,
            save_dir: os.PathLike) -> None:

        if self.algorithm.embed_dim == 2:
            self.fig.savefig(os.path.join(save_dir, self.FIG_FNAME))
        with open(os.path.join(save_dir, self.KNN_ACCS_FNAME), 'w') as f:
            json.dump({"knn_accs": self.knn_accs}, f, indent=4)

    @classmethod
    def load(
            cls: type[Self],
            save_dir: os.PathLike) -> ExperimentStats:

        with open(os.path.join(save_dir, cls.KNN_ACCS_FNAME), 'r') as f:
            knn_accs = json.load(f)["knn_accs"]
        stats = ExperimentStats(
            fig_fname=os.path.join(save_dir, cls.FIG_FNAME),
            knn_accs=knn_accs)

        return stats
