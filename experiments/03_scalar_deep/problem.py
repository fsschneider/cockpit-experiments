"""DeepOBS testproblem implementation for setting described in

- Mulayoff, R., & Michaeli, T.,
  Unique properties of wide minima in deep networks (2020)
- Ginsburg, B.,
  On regularization of gradient descent, layer imbalance and flat minima (2020).
"""

import warnings

import backobs
import deepobs
import numpy
import torch
from deepobs.pytorch.datasets.dataset import DataSet
from deepobs.pytorch.testproblems.testproblem import UnregularizedTestproblem


class BaseTestproblem(UnregularizedTestproblem):
    """Base class for all DeepOBS toy problems."""

    def get_batch_loss_and_accuracy_func(
        self, reduction="mean", add_regularization_if_available=True
    ):
        inputs, labels = self._get_next_batch()
        inputs = inputs.to(self._device)
        labels = labels.to(self._device)

        def forward_func():
            # in evaluation phase is no gradient needed
            if self.phase in ["train_eval", "test", "valid"]:
                with torch.no_grad():
                    outputs = self.net(inputs)
                    loss = self.loss_function(reduction=reduction)(outputs, labels)
            else:
                outputs = self.net(inputs)
                loss = self.loss_function(reduction=reduction)(outputs, labels)

            accuracy = 0.0

            if add_regularization_if_available:
                regularizer_loss = self.get_regularization_loss()
            else:
                regularizer_loss = torch.tensor(0.0, device=torch.device(self._device))

            return loss + regularizer_loss, accuracy

        return forward_func


class scalar_deep(BaseTestproblem):
    def __init__(self, batch_size, l2_reg=None, depth=2, sigma_xx=1, sigma_xy=1):
        """Deep scalar linear network.

        Note:
            The setting is described in Section 2 of
                - Ginsburg, B.,
                  On regularization of gradient descent, layer imbalance
                  and flat minima (2020).

        Args:
            depth (int): Number of layers.
            sigma_xx (float): Input variance.
            sigma_xy (float): Input-output covariance.
        """
        super().__init__(batch_size, l2_reg=l2_reg)
        self._depth = depth
        self._sigma_xx = sigma_xx
        self._sigma_xy = sigma_xy

    def set_up(self):
        """Set up the quadratic test problem."""
        self.net = self._make_net()
        self.data = self._make_data()
        self.net.to(self._device)
        self.loss_function = torch.nn.MSELoss
        self.regularization_groups = self.get_regularization_groups()

    def _make_net(self):
        layers = [torch.nn.Linear(1, 1, bias=False) for _ in range(self._depth)]
        assert self._depth == 2
        init_values = [0.1, 1.7]

        for idx, value in enumerate(init_values):
            layers[idx].weight.data[0] = value

        return torch.nn.Sequential(*layers)

    def _make_data(self):
        return scalar_deep_data(self._batch_size, self._sigma_xx, self._sigma_xy)


class scalar_deep_data(DataSet):
    def __init__(self, batch_size, sigma_xx, sigma_xy, train_size=100):
        self._name = "toy"
        self._sigma_xx = sigma_xx
        self._sigma_xy = sigma_xy
        self._train_size = train_size
        super().__init__(batch_size)

    def _make_data(self, seed=42):
        # train, valid, test
        num_sets = 3
        num_samples = num_sets * self._train_size

        rng = numpy.random.RandomState(seed)

        x = torch.from_numpy(numpy.float32(rng.normal(size=(num_samples, 1))))

        slope = 1.4
        noise_level = 1

        warnings.warn("Ignoring argument 'sigma_xx'")
        warnings.warn("Ignoring argument 'sigma_xy'")
        y = slope * x + noise_level * torch.from_numpy(
            numpy.float32(rng.normal(size=(num_samples, 1)))
        )

        # scale
        # x *= (self._sigma_xx / (x ** 2).mean()).sqrt()
        # y *= self._sigma_xy / (x * y).mean()

        # check
        # assert numpy.isclose((x ** 2).mean().item(), self._sigma_xx)
        # assert numpy.isclose((x * y).mean().item(), self._sigma_xy)

        # split
        x_train, x_valid, x_test = torch.split(x, self._train_size)
        y_train, y_valid, y_test = torch.split(y, self._train_size)

        return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

    def _make_train_and_valid_dataloader(self):
        (x_train, y_train), (x_valid, y_valid), _ = self._make_data()

        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        valid_dataset = torch.utils.data.TensorDataset(x_valid, y_valid)

        train_loader = self._make_dataloader(train_dataset, shuffle=True)
        valid_loader = self._make_dataloader(valid_dataset)

        return train_loader, valid_loader

    def _make_test_dataloader(self):
        _, _, (x_test, y_test) = self._make_data()
        test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

        return self._make_dataloader(test_dataset)

    def _make_train_eval_dataloader(self):
        return self._train_dataloader


def register():
    """Let DeepOBS and BackOBS know about the existence of the toy problem."""
    # DeepOBS
    deepobs.pytorch.testproblems.scalar_deep = scalar_deep

    # for CockpitPlotter
    if "scalar" in deepobs.config.DATA_SET_NAMING.keys():
        assert deepobs.config.DATA_SET_NAMING["scalar"] == "Scalar"
    else:
        deepobs.config.DATA_SET_NAMING["scalar"] = "Scalar"

    if "deep" in deepobs.config.TP_NAMING.keys():
        assert deepobs.config.TP_NAMING["deep"] == "Deep"
    else:
        deepobs.config.TP_NAMING["deep"] = "deep"

    # BackOBS
    backobs.utils.ALL += (scalar_deep,)
    backobs.utils.REGRESSION += (scalar_deep,)
    backobs.utils.SUPPORTED += (scalar_deep,)
    backobs.integration.SUPPORTED += (scalar_deep,)
