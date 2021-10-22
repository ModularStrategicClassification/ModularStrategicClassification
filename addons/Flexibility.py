from typing import Any, Callable, Dict, List, Optional

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np
from numpy import ndarray
import torch
from torch import tensor
from torch.nn.parameter import Parameter

import lib.CommonDefinitions as Commons
from lib.StrategicModel import StrategicModel
from lib.ResponseMapping import ResponseMapping


class LinearFlexibleResponseMapping(ResponseMapping):
    """
    The linear flexible response mapping class, implements the response mapping for a model whose cost function is linear and
    flexible, i.e. the weights in the cost function are also parameters which change between batches.
    """

    def __init__(self, x_dim_linear: int, batch_size: int, cost_const_kwargs: Optional[Dict[str, Any]] = None,
                 train_slope: float = Commons.TRAIN_SLOPE, eval_slope: float = Commons.EVAL_SLOPE,
                 x_lower_bound: float = Commons.X_LOWER_BOUND, x_upper_bound: float = Commons.X_UPPER_BOUND,
                 diff_threshold: float = Commons.DIFF_THRESHOLD, iteration_cap: int = Commons.ITERATION_CAP):
        """
        Initializes a linear flexible response mapping.
        :param x_dim_linear: The amount of features each user has to which the linear score function applies.
        :param batch_size: The batch size, the amount of users optimized at once.
        :param cost_const_kwargs: Constant keyword arguments which should be passed to the cost function on computation.
        :param train_slope: The slope of the sign approximation when training the model.
        :param eval_slope: The slope of the sign approximation when evaluating the model.
        :param x_lower_bound: The lower bound constraint of the optimization.
        :param x_upper_bound: The upper bound constraint of the optimization.
        :param diff_threshold: The threshold for the CCP procedure.
        :param iteration_cap: The maximal amount of iterations we allow the CCP to run.
        """
        super().__init__(x_dim_linear, batch_size, self.linear_cost_fn_dpp, self.linear_cost_fn_dpp_not_batched,
                         cost_const_kwargs, train_slope, eval_slope, x_lower_bound, x_upper_bound, diff_threshold,
                         iteration_cap)

    def create_optimization_problem(self) -> None:
        """
        Creates the non-differentiable optimization problem.
        """
        self.X = cp.Variable((self.batch_size, self.x_dim_linear))
        self.R = cp.Parameter((self.batch_size, self.x_dim_linear))
        self.w = cp.Parameter(self.x_dim_linear)
        self.b = cp.Parameter(1)
        self.nonlinear_score = cp.Parameter(self.batch_size)
        self.f_der = cp.Parameter((self.batch_size, self.x_dim_linear))

        self.v = cp.Parameter(self.x_dim_linear)
        self.Rv = cp.Parameter(self.batch_size)
        flexible_cost_kwargs = {"v": self.v, "Rv": self.Rv, **self.cost_const_kwargs}

        CCP_target_train = self.CCP_target_batched(self.X, self.R, self.w, self.b, self.nonlinear_score, self.f_der,
                                                   self.train_slope, flexible_cost_kwargs)
        CCP_target_eval = self.CCP_target_batched(self.X, self.R, self.w, self.b, self.nonlinear_score, self.f_der,
                                                  self.eval_slope, flexible_cost_kwargs)

        CCP_objective_train = cp.Maximize(cp.sum(CCP_target_train))
        CCP_objective_eval = cp.Maximize(cp.sum(CCP_target_eval))

        CCP_constraints = [self.X >= self.x_lower_bound, self.X <= self.x_upper_bound]

        CCP_problem_train = cp.Problem(CCP_objective_train, CCP_constraints)
        CCP_problem_eval = cp.Problem(CCP_objective_eval, CCP_constraints)

        self.CCP_problems = {self.train_slope: CCP_problem_train, self.eval_slope: CCP_problem_eval}

    def create_differential_optimization_problem(self) -> None:
        """
        Creates the differentiable optimization problem.
        """
        x = cp.Variable(self.x_dim_linear)
        r = cp.Parameter(self.x_dim_linear, value=np.random.randn(self.x_dim_linear))
        w = cp.Parameter(self.x_dim_linear, value=np.random.randn(self.x_dim_linear))
        b = cp.Parameter(1, value=np.random.randn(1))
        nonlinear_score = cp.Parameter(1, value=np.random.randn(1))
        f_der = cp.Parameter(self.x_dim_linear, value=np.random.randn(self.x_dim_linear))

        # The flexible cost parameters.
        v = cp.Parameter(self.x_dim_linear)
        rv = cp.Parameter(1)
        flexible_cost_kwargs = {"v": v, "rv": rv, **self.cost_const_kwargs}

        CCP_target_grad = self.CCP_target_not_batched(x, r, w, b, nonlinear_score, f_der, self.train_slope,
                                                      flexible_cost_kwargs)
        CCP_objective_grad = cp.Maximize(CCP_target_grad)
        CCP_constraints_grad = [x >= self.x_lower_bound, x <= self.x_upper_bound]
        CCP_problem_grad = cp.Problem(CCP_objective_grad, CCP_constraints_grad)
        self.CCP_layer = CvxpyLayer(CCP_problem_grad, parameters=[r, w, b, nonlinear_score, f_der, v, rv], variables=[x])

    def solve_optimization_problem(self, X: tensor, w: tensor, b: tensor, nonlinear_score: tensor, **kwargs) -> tensor:
        """
        Solves the non-differentiable optimization problem for the given X, w, b.
        :param X: The user's features, of size (self.batch_size, self.x_dim).
        :param w: The model's weights, of size (self.x_dim).
        :param b: The model's bias, of size (1)
        :param nonlinear_score: The nonlinear score of the user's features, of size (self.batch_size).
        :param kwargs: Any kwargs for the optimization problem. Should include the slope (train/eval slope).
        :return: The optimal features to which the users should move.
        """
        slope = kwargs.pop("slope")
        v = kwargs.pop("v")

        X_np = X.numpy()
        w_np = w.detach().numpy()
        b_np = b.detach().numpy()
        nonlinear_score_np = nonlinear_score.detach().numpy()
        v_np = v.detach().numpy()

        self.R.value = X_np
        self.w.value = w_np
        self.b.value = b_np
        self.nonlinear_score.value = nonlinear_score_np
        self.v.value = v_np
        self.Rv.value = X_np @ v_np

        # Set the initial value of the variable x to X, for the initial approximation location.
        self.X.value = X_np

        iteration = 0
        diff = np.inf
        while diff > self.diff_threshold and iteration < self.iteration_cap:
            iteration += 1
            approximation_X = self.X.value
            self.f_der.value = Commons.f_der_numpy(approximation_X, w_np, b_np, nonlinear_score_np, slope)
            self.CCP_problems[slope].solve()
            diff = np.linalg.norm(self.X.value - approximation_X) / self.batch_size

        return torch.from_numpy(self.X.value)

    def solve_differential_optimization_problem(self, X: tensor, w: tensor, b: tensor, nonlinear_score: tensor, X_opt: tensor,
                                                **kwargs) -> tensor:
        """
        Solves the differentiable optimization problem for the given X, w, b at X_opt.
        :param X: The user's features, of size (any_batch_size, self.x_dim).
        :param w: The model's weights, of size (self.x_dim).
        :param b: The model's bias, of size (1).
        :param nonlinear_score: The nonlinear score of the user's features, of size (any_batch_size).
        :param X_opt: The calculated optimal X by the non-differentiable problem, of size (any_batch_size, self.x_dim).
        :param kwargs: Any kwargs for the optimization problem.
        :return: The optimal features to which the users should move, with tracked gradients w.r.t w, b.
        """
        f_der = Commons.f_der_torch(X_opt, w, b, nonlinear_score, self.train_slope)
        nonlinear_score_batched = nonlinear_score.unsqueeze(1)
        v = kwargs.pop("v")
        rv = (X @ v).unsqueeze(1)
        return self.CCP_layer(X, w, b, nonlinear_score_batched, f_der, v, rv)[0]

    @staticmethod
    def linear_cost_fn_dpp(X: ndarray, R: ndarray, v: ndarray, Rv: ndarray, scale: float, epsilon: float) -> ndarray:
        """
        The linear cost function, in dpp form (so that the cvxpy can use it with a parametric weight vector v).
        """
        return scale * (epsilon * cp.square(cp.norm(X - R, 2, axis=1)) + (1 - epsilon) * cp.pos(X @ v - Rv))

    @staticmethod
    def linear_cost_fn_dpp_not_batched(x: ndarray, r: ndarray, v: ndarray, rv: ndarray, scale: float,
                                       epsilon: float) -> ndarray:
        """
        The linear cost function, not batched, in dpp form (so that the cvxpy can use it with a parametric weight vector v).
        """
        return scale * (epsilon * cp.sum_squares(x - r) + (1 - epsilon) * cp.pos(x @ v - rv))


class LinearFlexibleStrategicModel(StrategicModel):
    """
    The linear flexible strategic model class, implements the strategic model for a model whose cost function is linear and
    flexible, i.e. the weights in the cost function are learnable and change between iterations (using a price function for the
    price of changing the cost function).
    """
    def __init__(self, x_dim: int, batch_size: int, v_init: tensor,
                 price_fn: Optional[Callable] = None, price_const_kwargs: Optional[Dict[str, Any]] = None, reg: float = 1,
                 mixing_parameter: float = 0.05, cost_scale: float = 1, loss_fn: Callable = Commons.hinge_loss,
                 nonlinear_transformation_indices: Optional[List[int]] = None,
                 nonlinear_transformation_fn: Optional[Callable] = None, nonlinear_transformation_output_dimension: int = -1,
                 utility_reg: Optional[float] = None, burden_reg: Optional[float] = None, recourse_reg: Optional[float] = None,
                 social_measure_dict: Optional[Dict] = None, train_slope: float = Commons.TRAIN_SLOPE,
                 eval_slope: float = Commons.EVAL_SLOPE, x_lower_bound: float = Commons.X_LOWER_BOUND,
                 x_upper_bound: float = Commons.X_UPPER_BOUND, diff_threshold: float = Commons.DIFF_THRESHOLD,
                 iteration_cap: int = Commons.ITERATION_CAP, strategic: bool = True):
        """
        Initializes a linear flexible strategic model.
        :param x_dim: The amount of features each user has.
        :param batch_size: The batch size, the amount of users the model will be trained on each batch.
        :param v_init: The initial weight vector for the cost function.
        :param price_fn: The price function to change from an initial weight vector to another.
        :param price_const_kwargs: Constant keyword arguments which should be passed to the price function on computation.
        :param reg: The regularization factor for the price function in the loss function of the model.
        :param mixing_parameter: The mixing parameter for the quadratic cost in the linear flexible cost.
        :param cost_scale: The scale parameter for the cost function.
        :param loss_fn: The loss function.
        :param nonlinear_transformation_indices: The indices to which we apply the nonlinear transformation function.
        :param nonlinear_transformation_fn: The nonlinear transformation function.
        :param nonlinear_transformation_output_dimension: The output dimension of the nonlinear transformation function.
        :param utility_reg: Regularization for the utility social measure.
        :param burden_reg: Regularization for the burden social measure.
        :param recourse_reg: Regularization for the recourse social measure.
        :param social_measure_dict: A dictionary to add more social measures to the training of the model.
        :param train_slope: The slope of the sign approximation when training the model.
        :param eval_slope: The slope of the sign approximation when evaluating the model.
        :param x_lower_bound: The lower bound constraint of the optimization.
        :param x_upper_bound: The upper bound constraint of the optimization.
        :param diff_threshold: The threshold for the CCP procedure.
        :param iteration_cap: The maximal amount of iterations we allow the CCP to run.
        :param strategic: Whether the model is strategic (speculates gaming of the users).
        """
        # We pass StrategicModel's __init__ the regular linear cost functions just so cvxpy won't get upset, we override the
        # response mapping with a linear flexible response mapping afterwards so it has no effect.
        super().__init__(x_dim, batch_size, loss_fn, Commons.linear_cost_cvxpy_batched, Commons.linear_cost_cvxpy_not_batched,
                         Commons.linear_cost_torch, {"v": v_init, "scale": cost_scale, "epsilon": mixing_parameter},
                         nonlinear_transformation_indices, nonlinear_transformation_fn,
                         nonlinear_transformation_output_dimension, utility_reg, burden_reg, recourse_reg, social_measure_dict,
                         train_slope, eval_slope, x_lower_bound, x_upper_bound, diff_threshold, iteration_cap, strategic)

        if price_fn is None:
            price_fn = self.default_price_fn
        self.price_fn = price_fn
        if price_const_kwargs is None:
            price_const_kwargs = {}
        self.price_const_kwargs = price_const_kwargs
        self.reg = reg
        self.v_init = v_init
        self.v = Parameter(torch.clone(v_init))

        self.response_mapping = LinearFlexibleResponseMapping(self.x_dim_linear, batch_size,
                                                              {"scale": cost_scale, "epsilon": mixing_parameter}, train_slope,
                                                              eval_slope, x_lower_bound, x_upper_bound, diff_threshold,
                                                              iteration_cap)

    def optimize_X(self, X: tensor, requires_grad: bool = True) -> tensor:
        """
        Given the users' features X, returns the optimal X to which the users should move. Directly calls the response
        mapping's optimize_X method.
        :param X: The users' features.
        :param requires_grad: Whether the results should track gradients w.r.t the model's parameters.
        :return: The optimal response for X w.r.t the current model parameters.
        """
        nonlinear_score = self.nonlinear_score(X)
        X_opt_linear = self.response_mapping.optimize_X(X[:, self.linear_indices], self.w, self.b, nonlinear_score,
                                                        requires_grad=requires_grad, v=self.v)

        X_opt = torch.clone(X)
        X_opt[:, self.linear_indices] = X_opt_linear
        return X_opt

    def loss(self, Y: tensor, Y_pred: tensor, X: tensor, X_opt: tensor) -> tensor:
        """
        Calculates the loss of the model's prediction.
        :param Y: The real classification of the input.
        :param Y_pred: The model's classification of the input.
        :param X: The input.
        :param X_opt: The response of the input, as computed in the forward step.
        :return: The loss of the model's classification.
        """
        loss_val = super().loss(Y, Y_pred, X, X_opt)
        loss_val += self.reg * self.price_fn(self.v, self.v_init, **self.price_const_kwargs)
        return loss_val

    def get_batch_data_for_saving(self, epoch: int, batch: int, X: tensor, X_opt: tensor, Y: tensor,
                                  Y_pred: tensor) -> List[Dict[str, float]]:
        """
        Returns the batch data in the saving format (dictionary of entries to save for each row in X, X_opt, etc.).
        :param epoch: The epoch number.
        :param batch: The batch number.
        :param X: The users' features.
        :param X_opt: The users' responses.
        :param Y: The users' ground-truth classes.
        :param Y_pred: The model's prediction of the users' classes.
        :return: The data in the saving format.
        """
        w_dict = {f"w_{j}": self.w[j].detach().numpy() for j in range(len(self.w))}
        if self.nonlinear_transformation_fn is not None:
            w_nonlinear_dict = {f"w_nonlinear_{j}": self.w_nonlinear[j].detach().numpy() for j in range(len(self.w_nonlinear))}
        else:
            w_nonlinear_dict = {}

        v_dict = {f"v_{j}": self.v[j].detach().numpy() for j in range(len(self.v))}

        def x_dict(row):
            return {f"x_{j}": X[row][j].detach().numpy() for j in range(len(X[row]))}

        def x_opt_dict(row):
            return {f"x_opt_{j}": X_opt[row][j].detach().numpy() for j in range(len(X_opt[row]))}

        return [{"epoch": epoch, "batch": batch,
                 **w_dict, "b": self.b[0].detach().numpy(), **w_nonlinear_dict, **v_dict,
                 **x_dict(row), "y": Y[row].detach().numpy(),
                 **x_opt_dict(row), "y_pred": Y_pred[row].detach().numpy()} for row in range(len(X))]

    @staticmethod
    def default_price_fn(v: tensor, v_init: tensor) -> tensor:
        """
        The default price function for changing the cost weights.
        :param v: The current weight vector.
        :param v_init: The initial weight vector.
        :return: The price to change from v_init to v.
        """
        v_norm = torch.norm(v)
        v_init_norm = torch.norm(v_init)
        cos = (v @ v_init) / (v_norm * v_init_norm)
        return torch.abs(v_norm - v_init_norm) + torch.norm(cos - 1)
