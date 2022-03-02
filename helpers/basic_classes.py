import torch
from enum import Enum, auto
import os.path as osp
from typing import NamedTuple
from torch.nn import CrossEntropyLoss

from helpers.git_path import get_git_path
from helpers.vn_modelnet_dataloader import ModelNetDataLoader

MODELNET_NUM_POINTS = 1024
NUM_WORKERS = 6


class PoolType(Enum):
    """
    an object for the different datasets
    """
    NONE = auto()
    MEAN = auto()
    MAX = auto()

    @staticmethod
    def from_string(s):
        try:
            return PoolType[s]
        except KeyError:
            raise ValueError()


class KroneckerArgs(NamedTuple):
    """
        All hyper parameters for the kronecker product
    """
    n_neighbors: int
    dynamic_knn: bool


class ReLUArgs(NamedTuple):
    """
        All hyper parameters for VNLeakyReLU
    """
    add: bool
    eps: float
    share: bool
    negative_slope: float


class GeneralHeadArgs(NamedTuple):
    """
        All hyper parameters for the head
    """
    k: int
    in_channel: int
    add_linears: bool
    u_shape: bool
    z_align: bool
    pool_type: PoolType


class TrainerArgs(NamedTuple):
    """
        All hyper parameters for the trainer
    """
    epochs: int
    lr: float
    additive_noise: float
    scale_noise: float
    SO3_train: bool


class Task(Enum):
    """
    an object for the different datasets
    """
    Classification = auto()

    @staticmethod
    def from_string(s):
        try:
            return Task[s]
        except KeyError:
            raise ValueError()

    def get_loss(self):
        return CrossEntropyLoss()


class DataSet(Enum):
    """
    an object for the different datasets
    """
    ModelNet40 = auto()

    @staticmethod
    def from_string(s):
        try:
            return DataSet[s]
        except KeyError:
            raise ValueError()

    def load(self, z_align: bool):
        """
        Loads the dataset as a DataLoader object according to the batch_arguments
        :param z_align: bool
        :return: train_loader
        :return: test_loader
        :return: out_channel: int
        """
        dataset_folder_path = osp.join(get_git_path(), 'datasets')
        dataset_path = osp.join(dataset_folder_path, 'modelnet40_normal_resampled')
        if not osp.exists(dataset_path):
            exit('To use the ModelNet40VN dataset follow the README')

        train_dataset = ModelNetDataLoader(root=dataset_path, npoint=MODELNET_NUM_POINTS, split='train',
                                           normal_channel=False)
        test_dataset = ModelNetDataLoader(root=dataset_path, npoint=MODELNET_NUM_POINTS, split='test',
                                          normal_channel=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=NUM_WORKERS, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=NUM_WORKERS, shuffle=False)

        return train_loader, test_loader, 40

    def get_task(self) -> Task:
        return Task.Classification
