from typing import Dict, Union

import cvxpy as cp
from numpy import ndarray
import numpy as np
import torch
from torch import tensor

# Constants
TRAIN_SLOPE = 1
EVAL_SLOPE = 5
X_LOWER_BOUND = -10
X_UPPER_BOUND = 10
EPOCHS_WITHOUT_IMPROVEMENT_CAP = 4
DIFF_THRESHOLD = 0.001
ITERATION_CAP = 100


# Score function & derivative
def score(x: Union[tensor, ndarray], w: Union[tensor, ndarray], b: Union[tensor, ndarray]) -> Union[tensor, ndarray]:
    return x @ w + b


def score_der(x: Union[tensor, ndarray], w: Union[tensor, ndarray], b: Union[tensor, ndarray]) -> Union[tensor, ndarray]:
    return w


# Hinge loss
def hinge_loss(Y: tensor, Y_pred: tensor) -> tensor:
    return torch.mean(torch.clamp(1 - Y_pred * Y, min=0))


# Quadratic cost
def quad_cost_cvxpy_batched(X: ndarray, R: ndarray, scale: float = 1) -> ndarray:
    return scale * cp.square(cp.norm(X - R, 2, axis=1))


def quad_cost_cvxpy_not_batched(x: ndarray, r: ndarray, scale: float = 1) -> ndarray:
    return scale * cp.sum_squares(x - r)


def quad_cost_torch(X: tensor, R: tensor, scale: float = 1) -> torch:
    return scale * torch.sum((X - R) ** 2, dim=1)


# Linear cost
def linear_cost_cvxpy_batched(X: ndarray, R: ndarray, v: ndarray, epsilon: float = 0.05, scale: float = 1) -> ndarray:
    return scale * (epsilon * cp.square(cp.norm(X - R, 2, axis=1)) + (1 - epsilon) * cp.pos((X - R) @ v))


def linear_cost_cvxpy_not_batched(x: ndarray, r: ndarray, v: ndarray, epsilon: float = 0.05, scale: float = 1) -> ndarray:
    return scale * (epsilon * cp.sum_squares(x - r) + (1 - epsilon) * cp.pos((x - r) @ v))


def linear_cost_torch(X: tensor, R: tensor, v: tensor, epsilon: float = 0.05, scale: float = 1) -> torch:
    return scale * (epsilon * torch.sum((X - R) ** 2, dim=1) + (1 - epsilon) * torch.clamp((X - R) @ v, min=0))


recognized_cost_functions: Dict[str, Dict[str, any]] = {
    "quad": {
        "cvxpy_batched": quad_cost_cvxpy_batched,
        "cvxpy_not_batched": quad_cost_cvxpy_not_batched,
        "torch": quad_cost_torch,
        "required_kwargs": []
    },
    "linear": {
        "cvxpy_batched": linear_cost_cvxpy_batched,
        "cvxpy_not_batched": linear_cost_cvxpy_not_batched,
        "torch": linear_cost_torch,
        "required_kwargs": ["v"]
    }
}


# f, g, f_der - cvxpy versions, not batched
def f(x: ndarray, w: ndarray, b: ndarray, slope: float) -> ndarray:  # Unused.
    z = score(x, w, b)
    return 0.5 * cp.norm(cp.hstack([1, (slope * z + 1)]), 2)


def g(x: ndarray, w: ndarray, b: ndarray, slope: float) -> ndarray:
    z = score(x, w, b)
    return 0.5 * cp.norm(cp.hstack([1, (slope * z - 1)]), 2)


def f_der(x: ndarray, w: ndarray, b: ndarray, slope: float) -> ndarray:  # Unused.
    z = score(x, w, b)
    return 0.5 * cp.multiply(slope * ((slope * z + 1) / cp.sqrt((slope * z + 1) ** 2 + 1)),
                             score_der(x, w, b))


# f, g, f_der - cvxpy versions, batched
def f_batch(x: ndarray, w: ndarray, b: ndarray, slope: float) -> ndarray:  # Unused.
    z = score(x, w, b)
    return 0.5 * cp.norm(cp.vstack([np.ones(1, x.shape[0]), cp.reshape((slope * z + 1), (1, x.shape[0]))]), 2, axis=0)


def g_batch(x: ndarray, w: ndarray, b: ndarray, slope: float) -> ndarray:
    z = score(x, w, b)
    return 0.5 * cp.norm(cp.vstack([np.ones((1, x.shape[0])), cp.reshape((slope * z - 1), (1, x.shape[0]))]), 2, axis=0)


def f_der_batch(x: ndarray, w: ndarray, b: ndarray, slope: float) -> ndarray:  # Unused.
    z = score(x, w, b)
    der = 0.5 * slope * ((slope * z + 1) / cp.sqrt((slope * z + 1) ** 2 + 1))
    return cp.reshape(der, (der.shape[0], 1)) @ cp.reshape(score_der(x, w, b), (1, x.shape[1]))


# f, g, f_der - torch versions
def f_torch(x: tensor, w: tensor, b: tensor, slope: float) -> torch:  # Unused.
    z = score(x, w, b)
    return 0.5 * torch.sqrt((slope * z + 1) ** 2 + 1)


def g_torch(x: tensor, w: tensor, b: tensor, slope: float) -> torch:  # Unused.
    z = score(x, w, b)
    return 0.5 * torch.sqrt((slope * z - 1) ** 2 + 1)


def f_der_torch(x: tensor, w: tensor, b: tensor, slope: float) -> torch:
    z = score(x, w, b)
    der = 0.5 * slope * ((slope * z + 1) / torch.sqrt((slope * z + 1) ** 2 + 1))
    return der.unsqueeze(1) @ score_der(x, w, b).unsqueeze(0)


# f, g, f_der - numpy versions
def f_numpy(x: ndarray, w: ndarray, b: ndarray, slope: float) -> ndarray:  # Unused.
    z = score(x, w, b)
    return 0.5 * np.sqrt((slope * z + 1) ** 2 + 1)


def g_numpy(x: ndarray, w: ndarray, b: ndarray, slope: float) -> ndarray:  # Unused.
    z = score(x, w, b)
    return 0.5 * np.sqrt((slope * z - 1) ** 2 + 1)


def f_der_numpy(x: ndarray, w: ndarray, b: ndarray, slope: float) -> ndarray:
    z = score(x, w, b)
    der = 0.5 * slope * ((slope * z + 1) / np.sqrt((slope * z + 1) ** 2 + 1))
    return np.expand_dims(der, axis=1) @ np.expand_dims(score_der(x, w, b), axis=0)
