from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import torch
from torch import tensor

import lib.CommonDefinitions as Commons


class AbstractSocialMeasure(ABC):
    """
    A class for an abstract social measure.
    Every social measure should implement get_loss_term, so that the strategic model can add it to its loss calculation.
    The other methods are for data manipulation (saving/loading) and printing during the training process.
    """

    @abstractmethod
    def get_loss_term(self, Y: tensor, Y_pred: tensor, X: tensor, X_opt: tensor, w: tensor, b: tensor) -> tensor:
        """
        Get the loss term related to the social measure.
        :param Y: The ground-truth classes of the input.
        :param Y_pred: The model's prediction on the input.
        :param X: The input data.
        :param X_opt: The optimal response of the input, as calculated by the model.
        :param w: The model's weights.
        :param b: The model's bias.
        :return: The loss term related to the social measure.
        """
        pass

    def init_fit_train(self, epochs: int, verbose: Optional[str]) -> None:
        """
        Called at the beginning of the fit process.
        :param epochs: The amount of epochs.
        :param verbose: The verbose state of the model ("epochs", "batches" or None).
        """
        pass

    def init_fit_validation(self, verbose: Optional[str]) -> None:
        """
        Called at the beginning of the fit process, given that there is a validation set.
        :param verbose: The verbose state of the model ("epochs", "batches" or None).
        """
        pass

    def begin_epoch(self, epoch: int, validation: bool) -> None:
        """
        Called at the beginning of every epoch.
        :param epoch: The epoch number.
        :param validation: Whether there is a validation set.
        """
        pass

    def begin_batch(self, epoch: int, batch: int, Xbatch: tensor, Ybatch: tensor) -> None:
        """
        Called at the beginning of every batch.
        :param epoch: The epoch number.
        :param batch: The batch number.
        :param Xbatch: The input X for this batch.
        :param Ybatch: The input Y for this batch.
        """
        pass

    def end_batch(self, epoch: int, batch: int, Xbatch: tensor, Ybatch: tensor, Xbatch_opt: tensor, Ybatch_pred: tensor,
                  w: tensor, b: tensor) -> None:
        """
        Called at the end of every batch.
        :param epoch: The epoch number.
        :param batch: The batch number.
        :param Xbatch: The input X for this batch.
        :param Ybatch: The input Y for this batch.
        :param Xbatch_opt: The optimal response of the input X, as calculated by the model.
        :param Ybatch_pred: The model's prediction on the input.
        :param w: The model's weights.
        :param b: The model's bias.
        """
        pass

    def get_batch_info(self, epoch: int, batch: int) -> str:
        """
        Gets information about the social measure at the end of a batch. Used for printing, if verbose == "batches".
        :param epoch: The epoch number.
        :param batch: The batch number.
        :return: Information about the social measure at the end of a batch.
        """
        return ""

    def begin_validation(self, epoch: int) -> None:
        """
        Called at the beginning of a validation step, given that there is a validation set.
        :param epoch: The epoch number.
        """
        pass

    def begin_validation_batch(self, epoch: int, batch: int, Xbatch: tensor, Ybatch: tensor) -> None:
        """
        Called at the beginning of every batch in the validation step.
        :param epoch: The epoch number.
        :param batch: The batch number.
        :param Xbatch: The input X for this batch.
        :param Ybatch: The input Y for this batch.
        """
        pass

    def end_validation_batch(self, epoch: int, batch: int, Xbatch: tensor, Ybatch: tensor, Xbatch_opt: tensor,
                             Ybatch_pred: tensor, w: tensor, b: tensor) -> None:
        """
        Called at the end of every batch in the validation step.
        :param epoch: The epoch number.
        :param batch: The batch number.
        :param Xbatch: The input X for this batch.
        :param Ybatch: The input Y for this batch.
        :param Xbatch_opt: The optimal response of the input X, as calculated by the model.
        :param Ybatch_pred: The model's prediction on the input.
        :param w: The model's weights.
        :param b: The model's bias.
        """
        pass

    def get_validation_info(self, epoch: int) -> str:
        """
        Gets information about the social measure at the end of a validation step.
        Used for printing if verbose is "epochs" or "batches".
        :param epoch: The epoch number.
        :return: Information about the social measure at the end of a validation step.
        """
        return ""

    def end_epoch(self, epoch: int) -> None:
        """
        Called at the end of every epoch.
        :param epoch: The epoch number.
        """
        pass

    def save(self, path: str, model_name: str, print_message: bool) -> None:
        """
        Called whenever the model is saved.
        :param path: The path of the directory where the model is saved.
        :param model_name: The model's name.
        :param print_message: Whether to print a message when saving.
        """
        pass

    def load(self, path: str, model_name: str, print_message: bool) -> None:
        """
        Called whenever the model is loaded.
        :param path: The path of the directory from which the model is loaded.
        :param model_name: the model's name.
        :param print_message: Whether to print a message when loading.
        """
        pass


class Utility(AbstractSocialMeasure):
    """
    The utility social measure.
    Measures the expected utility of the users, i.e. the expectation over (gain - lost) of the users.
    """

    def __init__(self, reg: float, cost_fn_torch: Callable, cost_const_kwargs: Dict[str, Any] = None,
                 train_slope: float = Commons.TRAIN_SLOPE, eval_slope: float = Commons.EVAL_SLOPE):
        """
        Initializes the utility social measure.
        :param reg: The regularization factor in the loss term.
        :param cost_fn_torch: The gaming cost function (for batched user data, torch version).
        :param cost_const_kwargs: Constant keyword arguments which should be passed to the cost function on computation.
        :param train_slope: The slope of the sign approximation when training the model.
        :param eval_slope: The slope of the sign approximation when evaluating the model.
        """
        self.reg = reg
        self.train_slope = train_slope
        self.eval_slope = eval_slope

        if cost_fn_torch is None:
            error_message = "User must supply a PyTorch version of the cost function to use the utility social measure."
            raise ValueError(error_message)
        self.cost_fn = cost_fn_torch
        if cost_const_kwargs is None:
            cost_const_kwargs = {}
        self.cost_const_kwargs = cost_const_kwargs

    def calc_utility(self, X: tensor, X_opt: tensor, Y_pred: tensor, requires_grad: bool = False) -> tensor:
        """
        Calculates the expected utility of the given batch.
        :param X: The input batch.
        :param X_opt: The users' responses.
        :param Y_pred: The prediction of the model on the users.
        :param requires_grad: Whether the results should track gradients w.r.t. the model's parameters.
        :return: The expected utility of the batch.
        """
        slope = self.train_slope if requires_grad else self.eval_slope
        with torch.set_grad_enabled(requires_grad):
            gain = 0.5 * (torch.sqrt((slope * Y_pred + 1) ** 2 + 1) - torch.sqrt((slope * Y_pred - 1) ** 2 + 1))
            cost = self.cost_fn(X_opt, X, **self.cost_const_kwargs)
            utility = torch.mean(gain - cost)
        return utility

    def get_loss_term(self, Y: tensor, Y_pred: tensor, X: tensor, X_opt: tensor, w: tensor, b: tensor) -> tensor:
        return -self.reg * self.calc_utility(X, X_opt, Y_pred, requires_grad=True)

    def init_fit_train(self, epochs: int, verbose: Optional[str]) -> None:
        self.train_utilities = []

    def init_fit_validation(self, verbose: Optional[str]) -> None:
        self.validation_utilities = []

    def begin_epoch(self, epoch: int, validation: bool) -> None:
        self.train_utilities.append([])

    def end_batch(self, epoch: int, batch: int, Xbatch: tensor, Ybatch: tensor, Xbatch_opt: tensor, Ybatch_pred: tensor,
                  w: tensor, b: tensor) -> None:
        utility = self.calc_utility(Xbatch, Xbatch_opt, Ybatch_pred, requires_grad=False)
        self.train_utilities[epoch].append(utility)

    def get_batch_info(self, epoch: int, batch: int) -> str:
        return f"{self.train_utilities[epoch][batch]:3.5f}"

    def begin_validation(self, epoch: int) -> None:
        self.validation_utilities.append([])

    def end_validation_batch(self, epoch: int, batch: int, Xbatch: tensor, Ybatch: tensor, Xbatch_opt: tensor,
                             Ybatch_pred: tensor, w: tensor, b: tensor) -> None:
        utility = self.calc_utility(Xbatch, Xbatch_opt, Ybatch_pred, requires_grad=False)
        self.validation_utilities[epoch].append(utility)

    def get_validation_info(self, epoch: int) -> str:
        return f"{np.mean(self.validation_utilities[epoch]):3.5f}"


class Burden(AbstractSocialMeasure):
    """
    The burden social measure.
    Measures the minimum cost a positively-labeled user must incur in order to be classified correctly.
    """

    def __init__(self, reg: float, x_dim: tensor,
                 cost_fn_not_batched: Callable, cost_fn_torch: Callable, cost_const_kwargs: Dict[str, Any] = None,
                 x_lower_bound: float = Commons.X_LOWER_BOUND, x_upper_bound: float = Commons.X_UPPER_BOUND):
        """
        Initializes the burden social measure.
        :param reg: The regularization factor in the loss term.
        :param x_dim: The amount of features each user has.
        :param cost_fn_not_batched: The gaming cost function (for non-batched user data, cvxpy version).
        :param cost_fn_torch: The gaming cost function (for batched user data, torch version).
        :param cost_const_kwargs: Constant keyword arguments which should be passed to the cost function on computation.
        :param x_lower_bound: The lower bound constraint of the optimization.
        :param x_upper_bound: The upper bound constraint of the optimization.
        """
        self.reg = reg

        if cost_fn_torch is None:
            error_message = "User must supply a PyTorch version of the cost function to use the burden social measure."
            raise ValueError(error_message)
        self.cost_fn_torch = cost_fn_torch
        if cost_const_kwargs is None:
            cost_const_kwargs = {}
        self.cost_const_kwargs = cost_const_kwargs

        x = cp.Variable(x_dim)
        r = cp.Parameter(x_dim, value=np.random.randn(x_dim))
        w = cp.Parameter(x_dim, value=np.random.randn(x_dim))
        b = cp.Parameter(1, value=np.random.randn(1))

        target = cost_fn_not_batched(x, r, **cost_const_kwargs)
        constraints = [Commons.score(x, w, b) >= 0, x >= x_lower_bound, x <= x_upper_bound]

        objective = cp.Minimize(target)
        problem = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(problem, parameters=[r, w, b], variables=[x])

    def calc_burden(self, X: tensor, Y: tensor, w: tensor, b: tensor, requires_grad: bool = False) -> tensor:
        """
        Calculates the burden of the given batch.
        :param X: The input batch.
        :param Y: The input batch's ground-truth labels.
        :param w: The model's weights.
        :param b: The model's bias.
        :param requires_grad: Whether the results should track gradients w.r.t. the model's parameters.
        :return: The burden of the batch.
        """
        with torch.set_grad_enabled(requires_grad):
            X_pos = X[Y == 1]
            if len(X_pos) == 0:
                return 0
            # X_min = argmin_{X' : score(X') >= 0} {cost(X', X_pos)}
            X_min = self.layer(X_pos, w, b)[0]
            # cost(X_min, X_pos) = min_{X' : score(X') >= 0} {cost(X, X_pos)}
            burden = torch.mean(self.cost_fn_torch(X_min, X_pos, **self.cost_const_kwargs))
        return burden

    def get_loss_term(self, Y: tensor, Y_pred: tensor, X: tensor, X_opt: tensor, w: tensor, b: tensor) -> tensor:
        return self.reg * self.calc_burden(X, Y, w, b, requires_grad=True)

    def init_fit_train(self, epochs: int, verbose: Optional[str]) -> None:
        self.train_burdens = []

    def init_fit_validation(self, verbose: Optional[str]) -> None:
        self.validation_burdens = []

    def begin_epoch(self, epoch: int, validation: bool) -> None:
        self.train_burdens.append([])

    def end_batch(self, epoch: int, batch: int, Xbatch: tensor, Ybatch: tensor, Xbatch_opt: tensor, Ybatch_pred: tensor,
                  w: tensor, b: tensor) -> None:
        burden = self.calc_burden(Xbatch, Ybatch, w, b, requires_grad=False)
        self.train_burdens[epoch].append(burden)

    def get_batch_info(self, epoch: int, batch: int) -> str:
        return f"{self.train_burdens[epoch][batch]:3.5f}"

    def begin_validation(self, epoch: int) -> None:
        self.validation_burdens.append([])

    def end_validation_batch(self, epoch: int, batch: int, Xbatch: tensor, Ybatch: tensor, Xbatch_opt: tensor,
                             Ybatch_pred: tensor, w: tensor, b: tensor) -> None:
        burden = self.calc_burden(Xbatch, Ybatch, w, b, requires_grad=False)
        self.validation_burdens[epoch].append(burden)

    def get_validation_info(self, epoch: int) -> str:
        return f"{np.mean(self.validation_burdens[epoch]):3.5f}"


class Recourse(AbstractSocialMeasure):
    """
    The recourse social measure.
    Measures the capacity of a user who is classified negatively to restore appeal through low-cost feature modification.
    """

    def __init__(self, reg: float):
        """
        Initializes the recourse social measure.
        :param reg: The regularization factor in the loss term.
        """
        self.reg = reg

    def calc_recourse(self, X: tensor, X_opt: tensor, w: tensor, b: tensor, requires_grad: bool = False) -> tensor:
        """
        Calculates the recourse of the given batch.
        :param X: The input batch.
        :param X_opt: The users' responses.
        :param w: The model's weights.
        :param b: The model's bias.
        :param requires_grad: Whether the results should track gradients w.r.t. the model's parameters.
        :return: The recourse of the batch.
        """
        with torch.set_grad_enabled(requires_grad):
            sigmoid = torch.nn.Sigmoid()

            original_score = Commons.score(X, w, b)
            is_neg = sigmoid(-original_score)

            opt_score = Commons.score(X_opt, w, b)
            is_not_able_to_be_pos = sigmoid(-opt_score)

            recourse = 1 - torch.mean(is_neg * is_not_able_to_be_pos)
        return recourse

    def get_loss_term(self, Y: tensor, Y_pred: tensor, X: tensor, X_opt: tensor, w: tensor, b: tensor) -> tensor:
        return self.reg * (1 - self.calc_recourse(X, X_opt, w, b, requires_grad=True))

    def init_fit_train(self, epochs: int, verbose: Optional[str]) -> None:
        self.train_recourses = []

    def init_fit_validation(self, verbose: Optional[str]) -> None:
        self.validation_recourses = []

    def begin_epoch(self, epoch: int, validation: bool) -> None:
        self.train_recourses.append([])

    def end_batch(self, epoch: int, batch: int, Xbatch: tensor, Ybatch: tensor, Xbatch_opt: tensor, Ybatch_pred: tensor,
                  w: tensor, b: tensor) -> None:
        recourse = self.calc_recourse(Xbatch, Xbatch_opt, w, b, requires_grad=False)
        self.train_recourses[epoch].append(recourse)

    def get_batch_info(self, epoch: int, batch: int) -> str:
        return f"{self.train_recourses[epoch][batch]:3.5f}"

    def begin_validation(self, epoch: int) -> None:
        self.validation_recourses.append([])

    def end_validation_batch(self, epoch: int, batch: int, Xbatch: tensor, Ybatch: tensor, Xbatch_opt: tensor,
                             Ybatch_pred: tensor, w: tensor, b: tensor) -> None:
        recourse = self.calc_recourse(Xbatch, Xbatch_opt, w, b, requires_grad=False)
        self.validation_recourses[epoch].append(recourse)

    def get_validation_info(self, epoch: int) -> str:
        return f"{np.mean(self.validation_recourses[epoch]):3.5f}"
