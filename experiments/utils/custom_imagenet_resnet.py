"""ImageNet-type data and modified ResNet50 without batch norm."""

import os
import sys

import torch
import torchvision
from deepobs import config
from deepobs.pytorch.datasets.dataset import DataSet
from deepobs.pytorch.testproblems.testproblem import UnregularizedTestproblem
from torch.utils import data as dat

sys.path.append(os.getcwd())
from experiments.utils.deepobs_runner import register, replace  # noqa


def classification_targets(N, num_classes):
    """Create random targets for classes 0, ..., `num_classes - 1`."""
    return torch.randint(size=(N,), low=0, high=num_classes)


class dummyimagenet(DataSet):
    """Random numbers and labels of ImageNet shape."""

    def __init__(self, batch_size, train_size=100):
        self._name = "dummyimagenet"
        self._train_size = train_size
        super().__init__(batch_size)

    def _make_labels(self):
        imagenet_classes = 1000
        return classification_targets(self._train_size, imagenet_classes)

    def _make_data(self):
        imagenet_shape = (3, 224, 224)
        return torch.rand(size=(self._train_size, *imagenet_shape))

    def _make_train_and_valid_dataloader(self):
        X = self._make_data()
        Y = self._make_labels()
        train_dataset = dat.TensorDataset(X, Y)

        X = self._make_data()
        Y = self._make_labels()
        valid_dataset = dat.TensorDataset(X, Y)

        train_loader = self._make_dataloader(train_dataset, shuffle=True)
        valid_loader = self._make_dataloader(valid_dataset)
        return train_loader, valid_loader

    def _make_test_dataloader(self):
        X = self._make_data()
        Y = self._make_labels()
        test_dataset = dat.TensorDataset(X, Y)
        return self._make_dataloader(test_dataset)

    def _make_train_eval_dataloader(self):
        return self._train_dataloader


def resnet50nobn():
    """Return ResNet50 with batch normalization replaced by identities."""

    def trigger(module):
        return isinstance(module, torch.nn.BatchNorm2d)

    def make_new(module):
        return torch.nn.Identity()

    model = torchvision.models.resnet50()
    replace(model, trigger, make_new)

    return model


class dummyimagenet_resnet50nobn(UnregularizedTestproblem):
    """ResNet50 without batch normalization, trained with dummy ImageNet-shaped data."""

    def __init__(self, batch_size, l2_reg=None):
        super().__init__(batch_size, l2_reg)

    def set_up(self):
        self.net = resnet50nobn()
        self.data = dummyimagenet(self._batch_size)
        self.net.to(self._device)
        self.loss_function = torch.nn.CrossEntropyLoss
        self.regularization_groups = self.get_regularization_groups()


def register_imagenet_problem():
    register(dummyimagenet_resnet50nobn)

    # set default parameters
    config.DEFAULT_TEST_PROBLEMS_SETTINGS["dummyimagenet_resnet50nobn"] = {
        "batch_size": 8,
        "num_epochs": 100,
    }
