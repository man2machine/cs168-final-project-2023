# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 01:58:05 2023

@author: Shahir
"""

import os
import abc
import json
import hashlib
import shutil
from dataclasses import dataclass, field

from enum import Enum
from typing import Self, Optional, Any, Union

import numpy as np
import numpy.typing as npt

import torch
from torch.utils.data import Dataset, Subset

from sklearn.model_selection import train_test_split

from cs168_project.utils import JSONDictSerializable, get_rel_pkg_path
from cs168_project.datasets import (
    DatasetType, get_img_dataset_raw, get_img_dataset_transforms, TransformDataset, transform_with_label,
    generate_sklearn_classification_dataset, get_sklearn_preset_dataset, DATASET_TO_NUM_SAMPLES, IMAGE_TORCH_DATASETS,
    DEFAULT_DATA_DIR)
from cs168_project.algorithm import (
    AlgorithmType, BaseReducerAlgorithm, PCAAlgorithm, TSNEAlgorithm, LLEAlgorithm, UMAPAlgorithm,
    AEAlgorithm, AEUMAPAlgorithm)
from cs168_project.evaluation import ExperimentStatsRunner, ExperimentStats

DEFAULT_EXPERIMENTS_DIR = get_rel_pkg_path("experiment_cache/")
os.makedirs(DEFAULT_EXPERIMENTS_DIR, exist_ok=True)

@dataclass(kw_only=True)
class BaseDatasetConfig(JSONDictSerializable):
    dataset_type: DatasetType

    @property
    @abc.abstractmethod
    def num_samples(
            self: Self) -> int:

        pass

    @abc.abstractmethod
    def generate_dataset(
            self: Self) -> Dataset:

        pass


@dataclass(kw_only=True)
class ImageDatasetConfig(BaseDatasetConfig):
    flatten: bool = False
    num_samples: int = field(init=False, default=0)
    num_samples_limit: Optional[int] = None
    seed: int = 0

    def __post_init__(
            self: Self):

        self.num_samples = DATASET_TO_NUM_SAMPLES[self.dataset_type]
        if self.num_samples_limit is not None:
            assert self.num_samples_limit > 200
            assert self.num_samples_limit < self.num_samples
        assert self.dataset_type in IMAGE_TORCH_DATASETS

    def to_dict(
            self: Self) -> dict[str, Any]:

        data = {
            'dataset_type': self.dataset_type,
            'num_samples_limit': self.num_samples_limit,
            'flatten': self.flatten,
        }

        return data

    @classmethod
    def from_dict(
            cls: type[Self],
            data: dict[str, Any]) -> Self:

        config = cls(
            dataset_type=data['dataset_type'],
            num_samples_limit=data['num_samples_limit'],
            flatten=data['flatten'])

        return config

    def generate_dataset(
            self: Self,
            data_dir: os.PathLike = DEFAULT_DATA_DIR) -> Dataset[tuple[torch.Tensor, int]]:

        dataset = get_img_dataset_raw(
            data_dir=data_dir,
            dataset_type=self.dataset_type,
            combine_train_test=True)['all']
        if self.num_samples_limit:
            labels = np.array([y for x, y in dataset], dtype=np.int64)
            data_indices, _ = train_test_split(
                np.arange(len(dataset)),
                train_size=self.num_samples_limit,
                random_state=self.seed,
                stratify=labels)
            dataset = Subset(dataset, data_indices)
        transform = transform_with_label(get_img_dataset_transforms(
            dataset_type=self.dataset_type)['test'])
        dataset = TransformDataset(dataset, transform)

        return dataset


@dataclass(kw_only=True)
class SKLearnCustomDatasetConfig(BaseDatasetConfig):
    num_samples: int = 10000
    num_features: int = 16
    num_informative: int = 10
    num_redundant: int = 6
    num_repeated: int = 0
    num_classes: int = 5
    num_clusters_per_class: int = 1
    seed: int = 0

    def __post_init__(
            self: Self):

        assert self.dataset_type == DatasetType.SKLEARN_CUSTOM
        assert self.num_samples >= 1000

    def to_dict(
            self: Self) -> dict[str, Any]:

        data = {
            'dataset_type': self.dataset_type,
            'num_samples': self.num_samples,
            'num_features': self.num_features,
            'num_informative': self.num_informative,
            'num_redundant': self.num_redundant,
            'num_repeated': self.num_repeated,
            'num_classes': self.num_classes,
            'num_clusters_per_class': self.num_clusters_per_class,
            'seed': self.seed
        }

        return data

    @classmethod
    def from_dict(
            cls: type[Self],
            data: dict[str, Any]):

        config = cls(
            dataset_type=data['dataset_type'],
            num_samples=data['num_samples'],
            num_features=data['num_features'],
            num_informative=data['num_informative'],
            num_redundant=data['num_redundant'],
            num_repeated=data['num_repeated'],
            num_classes=data['num_classes'],
            num_clusters_per_class=data['num_clusters_per_class'],
            seed=data['seed'])

        return config

    def generate_dataset(
            self: Self) -> Dataset[tuple[npt.NDArray[np.float64], int]]:

        dataset = generate_sklearn_classification_dataset(
            num_samples=self.num_samples,
            num_features=self.num_features,
            num_informative=self.num_informative,
            num_redundant=self.num_redundant,
            num_repeated=self.num_repeated,
            num_classes=self.num_classes,
            num_clusters_per_class=self.num_clusters_per_class,
            seed=self.seed)

        return dataset


@dataclass(kw_only=True)
class SKLearnPresetDatasetConfig(BaseDatasetConfig):
    num_samples: int = field(init=False, default=0)
    num_samples_limit: Optional[int] = None

    def __post_init__(
            self: Self):

        self.num_samples = DATASET_TO_NUM_SAMPLES[self.dataset_type]
        assert self.dataset_type in (DatasetType.SKLEARN_IRIS, DatasetType.SKLEARN_DIGITS)

    def to_dict(
            self: Self) -> dict[str, Any]:

        data = {
            'dataset_type': self.dataset_type,
            'num_samples_limit': self.num_samples_limit,
        }

        return data

    @classmethod
    def from_dict(
            cls: type[Self],
            data: dict[str, Any]) -> Self:

        config = cls(
            dataset_type=data['dataset_type'],
            num_samples_limit=data['num_samples_limit'])

        return config

    def generate_dataset(
            self) -> Dataset[tuple[npt.NDArray[np.float64], int]]:

        dataset = get_sklearn_preset_dataset(
            dataset_type=self.dataset_type)

        return dataset


@dataclass(kw_only=True)
class BaseAlgorithmConfig(JSONDictSerializable):
    seed: Optional[int] = None
    embed_dim: int = 2

    @property
    @abc.abstractmethod
    def algorithm_type(
            self: Self) -> AlgorithmType:

        pass

    @abc.abstractmethod
    def generate_algorithm(
            self: Self,
            dataset_type: DatasetType,
            dataset: Dataset) -> BaseReducerAlgorithm:

        pass


@dataclass(kw_only=True)
class PCAAlgorithmConfig(BaseAlgorithmConfig):
    algorithm_type: AlgorithmType = field(init=False, default=AlgorithmType.PCA)

    def to_dict(
            self: Self) -> dict[str, Any]:

        data = {
            'algorithm_type': self.algorithm_type,
            'seed': self.seed,
            'embed_dim': self.embed_dim
        }

        return data

    @classmethod
    def from_dict(
            cls: type[Self],
            data: dict[str, Any]) -> Self:

        config = cls(
            seed=data['seed'],
            embed_dim=data['embed_dim'])

        return config

    def generate_algorithm(
            self: Self,
            dataset_type: DatasetType,
            dataset: Dataset) -> PCAAlgorithm:

        alg = PCAAlgorithm(
            dataset_type=dataset_type,
            dataset=dataset,
            embed_dim=self.embed_dim)

        return alg


@dataclass(kw_only=True)
class TSNEAlgorithmConfig(BaseAlgorithmConfig):
    algorithm_type: AlgorithmType = field(init=False, default=AlgorithmType.TSNE)
    perplexity: int = 50

    def to_dict(
            self: Self) -> dict[str, Any]:

        data = {
            'algorithm_type': self.algorithm_type,
            'seed': self.seed,
            'embed_dim': self.embed_dim,
            'perplexity': self.perplexity
        }

        return data

    @classmethod
    def from_dict(
            cls: type[Self],
            data: dict[str, Any]) -> Self:

        config = cls(
            seed=data['seed'],
            embed_dim=data['embed_dim'],
            perplexity=data['perplexity'])

        return config

    def generate_algorithm(
            self: Self,
            dataset_type: DatasetType,
            dataset: Dataset) -> TSNEAlgorithm:

        alg = TSNEAlgorithm(
            dataset_type=dataset_type,
            dataset=dataset,
            embed_dim=self.embed_dim,
            seed=self.seed,
            perplexity=self.perplexity)

        return alg


@dataclass(kw_only=True)
class LLEAlgorithmConfig(BaseAlgorithmConfig):
    algorithm_type: AlgorithmType = field(init=False, default=AlgorithmType.LLE)

    def to_dict(
            self: Self) -> dict[str, Any]:

        data = {
            'algorithm_type': self.algorithm_type,
            'seed': self.seed,
            'embed_dim': self.embed_dim
        }

        return data

    @classmethod
    def from_dict(
            cls: type[Self],
            data: dict[str, Any]) -> Self:

        config = cls(
            seed=data['seed'],
            embed_dim=data['embed_dim'])

        return config

    def generate_algorithm(
            self: Self,
            dataset_type: DatasetType,
            dataset: Dataset) -> LLEAlgorithm:

        alg = LLEAlgorithm(
            dataset_type=dataset_type,
            dataset=dataset,
            embed_dim=self.embed_dim,
            seed=self.seed)

        return alg


@dataclass(kw_only=True)
class UMAPAlgorithmConfig(BaseAlgorithmConfig):
    algorithm_type: AlgorithmType = field(init=False, default=AlgorithmType.UMAP)

    def to_dict(
            self: Self) -> dict[str, Any]:

        data = {
            'algorithm_type': self.algorithm_type,
            'seed': self.seed,
            'embed_dim': self.embed_dim
        }

        return data

    @classmethod
    def from_dict(
            cls: type[Self],
            data: dict[str, Any]) -> Self:

        config = cls(
            seed=data['seed'],
            embed_dim=data['embed_dim'])

        return config

    def generate_algorithm(
            self: Self,
            dataset_type: DatasetType,
            dataset: Dataset) -> UMAPAlgorithm:

        alg = UMAPAlgorithm(
            dataset_type=dataset_type,
            dataset=dataset,
            embed_dim=self.embed_dim,
            seed=self.seed)

        return alg


@dataclass(kw_only=True)
class AEAlgorithmConfig(JSONDictSerializable):
    pass

@dataclass(kw_only=True)
class ExperimentSetup:
    dataset: Dataset
    algorithm: BaseReducerAlgorithm
    stats_runner: ExperimentStatsRunner


@dataclass(kw_only=True)
class ExperimentConfig(JSONDictSerializable):
    def __init__(
        self: Self,
            dataset_config: BaseDatasetConfig,
            algorithm_config: BaseAlgorithmConfig,
            trial_index: Optional[int] = None) -> None:

        self.dataset_config = dataset_config
        self.algorithm_config = algorithm_config
        self.trial_index = trial_index

    def to_dict(
            self: Self) -> dict[str, Any]:

        data = {
            'dataset_config': self.dataset_config.to_dict(),
            'algorithm_config': self.algorithm_config.to_dict(),
            'trial_index': self.trial_index
        }

        return data

    @classmethod
    def from_dict(
            cls: type[Self],
            data: dict[str, Any]) -> Self:

        dataset_type = DatasetType(data['dataset_config']['dataset_type'])
        match dataset_type:
            case DatasetType.MNIST | DatasetType.FASHION_MNIST | DatasetType.CIFAR10 | DatasetType.CIFAR100:
                dataset_config = ImageDatasetConfig.from_dict(data['dataset_config'])
            case DatasetType.SKLEARN_IRIS | DatasetType.SKLEARN_DIGITS:
                dataset_config = SKLearnPresetDatasetConfig.from_dict(data['dataset_config'])
            case DatasetType.SKLEARN_CUSTOM:
                dataset_config = SKLearnCustomDatasetConfig.from_dict(data['dataset_config'])
            case _:
                raise ValueError()

        algorithm_type = AlgorithmType(data['algorithm_config']['algorithm_type'])
        match algorithm_type:
            case AlgorithmType.PCA:
                algorithm_config = PCAAlgorithmConfig.from_dict(data['algorithm_config'])
            case AlgorithmType.TSNE:
                algorithm_config = TSNEAlgorithmConfig.from_dict(data['algorithm_config'])
            case AlgorithmType.LLE:
                algorithm_config = LLEAlgorithmConfig.from_dict(data['algorithm_config'])
            case AlgorithmType.UMAP:
                algorithm_config = UMAPAlgorithmConfig.from_dict(data['algorithm_config'])
            case AlgorithmType.AE:
                algorithm_config = AEAlgorithmConfig.from_dict(data['algorithm_config'])
            case _:
                raise ValueError()

        config = cls(
            dataset_config=dataset_config,
            algorithm_config=algorithm_config,
            trial_index=data['trial_index'])

        return config

    def generate_setup(
            self: Self) -> ExperimentSetup:

        dataset = self.dataset_config.generate_dataset()
        algorithm = self.algorithm_config.generate_algorithm(
            dataset_type=self.dataset_config.dataset_type,
            dataset=dataset)
        stats_runner = ExperimentStatsRunner(
            dataset_type=self.dataset_config.dataset_type,
            dataset=dataset,
            algorithm=algorithm)
        setup = ExperimentSetup(
            dataset=dataset,
            algorithm=algorithm,
            stats_runner=stats_runner)

        return setup


@dataclass(kw_only=True)
class ExperimentState(JSONDictSerializable):
    def __init__(
            self,
            run_complete=False,
            stats_complete=False) -> None:

        self.alg_complete = run_complete
        self.stats_complete = stats_complete

    def to_dict(
            self: Self) -> dict[str, Any]:

        data = {
            'alg_complete': self.alg_complete,
            'stats_complete': self.stats_complete
        }

        return data

    @classmethod
    def from_dict(
            cls: type[Self],
            data: dict[str, Any]) -> Self:

        state = cls(
            run_complete=data['alg_complete'],
            stats_complete=data['stats_complete'])

        return state


class ExperimentManager:
    ALGORITHM_SUBDIR = "alg"
    STATS_SUBDIR = "stats"

    def __init__(
            self: Self,
            data_dir: os.PathLike = DEFAULT_DATA_DIR,
            experiment_dir: os.PathLike = DEFAULT_EXPERIMENTS_DIR) -> None:

        self.data_dir = data_dir
        self.experiment_dir = experiment_dir
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        self._load_index()

    def _load_index(
            self: Self) -> None:

        index_fname = os.path.join(self.experiment_dir, "index.json")
        if os.path.exists(index_fname):
            with open(index_fname, 'r') as f:
                index = json.load(f)
            self.index = index
        else:
            self.index = {}
            self._save_index()

    def _save_index(
            self: Self) -> None:

        index_fname = os.path.join(self.experiment_dir, "index.json")
        with open(index_fname, 'w') as f:
            json.dump(self.index, f, indent=4)

    def _save_config(
            self: Self,
            config: ExperimentConfig,
            workspace_dir: os.PathLike) -> None:

        config_fname = os.path.join(workspace_dir, "config.json")
        with open(config_fname, 'w') as f:
            json.dump(config.to_dict(), f, indent=4)

    def _load_state(
            self: Self,
            workspace_dir: os.PathLike) -> ExperimentState:

        state_fname = os.path.join(workspace_dir, "state.json")
        with open(state_fname, 'r') as f:
            state = json.load(f)
        state = ExperimentState.from_dict(state)

        return state

    def _save_state(
            self: Self,
            state: ExperimentState,
            workspace_dir: os.PathLike) -> None:

        state_fname = os.path.join(workspace_dir, "state.json")
        with open(state_fname, 'w') as f:
            json.dump(state.to_dict(), f, indent=4)

    def _get_workspace_dir_from_hash(
            self: Self,
            key: str):

        return os.path.join(self.experiment_dir, key)

    def get_workspace_dir(
            self: Self,
            config: ExperimentConfig) -> os.PathLike:

        key = self.find_experiment(config)
        if not key:
            raise ValueError("Experiment not found")
        return self._get_workspace_dir_from_hash(key)

    def find_experiment(
            self: Self,
            config: ExperimentConfig) -> Union[bool, str]:

        config_bytes = config.to_bytes()
        hasher = hashlib.blake2b(digest_size=8)
        hasher.update(config_bytes)
        key = hasher.hexdigest()
        if key in self.index:
            return key
        else:
            for key, c in self.index.items():
                if c == config.to_dict():
                    return key
            return False

    def add_experiment(
            self: Self,
            config: ExperimentConfig,
            exist_ok: bool = False) -> str:

        config_bytes = config.to_bytes()
        experiment_exists = self.find_experiment(config)
        if experiment_exists and exist_ok:
            return experiment_exists
        elif experiment_exists and not exist_ok:
            raise ValueError()
        num_salt = 0
        while True:
            hasher = hashlib.blake2b(digest_size=8)
            hasher.update(config_bytes)
            if num_salt:
                hasher.update('a' * num_salt)
            key = hasher.hexdigest()
            if key in self.index:
                num_salt += 1
                continue
            break
        self.index[key] = config.to_dict()
        self._save_index()
        workspace_dir = self._get_workspace_dir_from_hash(key)
        os.makedirs(workspace_dir)
        state = ExperimentState()
        self._save_config(config, workspace_dir)
        self._save_state(state, workspace_dir)

        return key

    def remove_experiment(
            self: Self,
            config: ExperimentConfig) -> None:

        key = self.find_experiment(config)
        self.index.pop(key)
        workspace_dir = self._get_workspace_dir_from_hash(key)
        shutil.rmtree(workspace_dir)

    def load_experiment(
            self: Self,
            config: ExperimentConfig,
            load_alg: bool = True) -> tuple[ExperimentSetup, ExperimentState]:

        workspace_dir = self.get_workspace_dir(config)
        setup = config.generate_setup()
        state = self._load_state(workspace_dir)
        if state.alg_complete and load_alg:
            setup.algorithm.load(os.path.join(workspace_dir, self.ALGORITHM_SUBDIR))

        return setup, state

    def load_stats(
            self: Self,
            config: ExperimentConfig) -> ExperimentStats:

        setup, state = self.load_experiment(config, load_alg=False)
        if not state.stats_complete:
            raise ValueError()
        workspace_dir = self.get_workspace_dir(config)
        stats = setup.stats_runner.load(os.path.join(workspace_dir, self.STATS_SUBDIR))

        return stats

    def run_algorithm(
            self: Self,
            config: ExperimentConfig,
            completed_ok: bool = False,
            overwrite: bool = False) -> None:

        workspace_dir = self.get_workspace_dir(config)
        state = self._load_state(workspace_dir)
        if state.alg_complete and not overwrite:
            if completed_ok:
                return
            else:
                raise ValueError("Algorithm already run")

        setup, state = self.load_experiment(config)
        alg_dir = os.path.join(workspace_dir, self.ALGORITHM_SUBDIR)
        os.makedirs(alg_dir, exist_ok=True)

        setup.algorithm.initialize()
        setup.algorithm.run()
        setup.algorithm.save(alg_dir)
        state.alg_complete = True
        self._save_state(state, workspace_dir)

    def run_stats(
            self: Self,
            config: ExperimentConfig,
            completed_ok: bool = False,
            overwrite: bool = False) -> None:

        workspace_dir = self.get_workspace_dir(config)
        state = self._load_state(workspace_dir)
        if state.stats_complete and not overwrite:
            if completed_ok:
                return
            else:
                raise ValueError("Algorithm already run")

        setup, state = self.load_experiment(config)
        stats_dir = os.path.join(workspace_dir, self.STATS_SUBDIR)
        os.makedirs(stats_dir, exist_ok=True)

        setup.stats_runner.run()
        setup.stats_runner.save(stats_dir)
        state.stats_complete = True
        self._save_state(state, workspace_dir)
