from typing import Any, Callable, Dict

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import torch
from torch import tensor
import torch.nn.functional as F

import lib.CommonDefinitions as Commons


class ResponseMapping:
    """
    The response mapping class, given the users' features and model parameters returns the optimal response of the
    users, i.e. the optimal features to which the users should move to maximize their utility (gain - cost).
    """

    def __init__(self, x_dim_linear: int, batch_size: int, cost_fn_batched: Callable = Commons.quad_cost_cvxpy_batched,
                 cost_fn_not_batched: Callable = Commons.quad_cost_cvxpy_not_batched,
                 cost_const_kwargs: Dict[str, Any] = None, train_slope: float = Commons.TRAIN_SLOPE,
                 eval_slope: float = Commons.EVAL_SLOPE, x_lower_bound: float = Commons.X_LOWER_BOUND,
                 x_upper_bound: float = Commons.X_UPPER_BOUND, diff_threshold: float = Commons.DIFF_THRESHOLD,
                 iteration_cap: int = Commons.ITERATION_CAP):
        """
        Initializes a response mapping.
        :param x_dim_linear: The amount of features each user has to which the linear score function applies.
        :param batch_size: The batch size, the amount of users optimized at once.
        :param cost_fn_batched: The cost function for batched user data.
        :param cost_fn_not_batched: The cost function for non-batched user data.
        :param cost_const_kwargs: Constant keyword arguments which should be passed to the cost function on computation.
        :param train_slope: The slope of the sign approximation when training the model.
        :param eval_slope: The slope of the sign approximation when evaluating the model.
        :param x_lower_bound: The lower bound constraint of the optimization.
        :param x_upper_bound: The upper bound constraint of the optimization.
        :param diff_threshold: The threshold for the CCP procedure.
        :param iteration_cap: The maximal amount of iterations we allow the CCP to run.
        """
        self.x_dim_linear = x_dim_linear
        self.batch_size = batch_size
        self.cost_batched = cost_fn_batched
        self.cost_not_batched = cost_fn_not_batched
        self.cost_const_kwargs = cost_const_kwargs
        self.train_slope = train_slope
        self.eval_slope = eval_slope
        self.x_lower_bound = x_lower_bound
        self.x_upper_bound = x_upper_bound
        self.diff_threshold = diff_threshold
        self.iteration_cap = iteration_cap

        self.create_optimization_problem()
        self.create_differential_optimization_problem()

    def create_optimization_problem(self) -> None:
        """
        Creates the non-differentiable optimization problem.
        The cvxpy solver handles batches only if we explicitly define the parameters and optimization variable with a
        batch dimension. Thus, the optimization problem uses CCP_target_batched with batched parameters.
        """
        self.X = cp.Variable((self.batch_size, self.x_dim_linear))
        self.R = cp.Parameter((self.batch_size, self.x_dim_linear))
        self.w = cp.Parameter(self.x_dim_linear)
        self.b = cp.Parameter(1)
        self.nonlinear_score = cp.Parameter(self.batch_size)
        self.f_der = cp.Parameter((self.batch_size, self.x_dim_linear))

        # Note: In order to make the optimization problem dpp, we separate it to two problems: one using train_slope
        # and one using eval_slope. One could define a cp.Parameter for the slope, but then in g_batch it would multiply
        # that parameter with the score, which includes parameters, thus making the problem not dpp.
        CCP_target_train = self.CCP_target_batched(self.X, self.R, self.w, self.b, self.nonlinear_score, self.f_der,
                                                   self.train_slope, self.cost_const_kwargs)
        CCP_target_eval = self.CCP_target_batched(self.X, self.R, self.w, self.b, self.nonlinear_score, self.f_der,
                                                  self.eval_slope, self.cost_const_kwargs)

        CCP_objective_train = cp.Maximize(cp.sum(CCP_target_train))
        CCP_objective_eval = cp.Maximize(cp.sum(CCP_target_eval))

        CCP_constraints = [self.X >= self.x_lower_bound, self.X <= self.x_upper_bound]

        CCP_problem_train = cp.Problem(CCP_objective_train, CCP_constraints)
        CCP_problem_eval = cp.Problem(CCP_objective_eval, CCP_constraints)

        self.CCP_problems = {self.train_slope: CCP_problem_train, self.eval_slope: CCP_problem_eval}

    def create_differential_optimization_problem(self) -> None:
        """
        Creates the differentiable optimization problem.
        The cvxpylayers solver handles batches by itself, so we don't need to explicitly state the batch dimension.
        Thus, the differential optimization problem uses CCP_target_not_batches with non-batched parameters.
        """
        x = cp.Variable(self.x_dim_linear)
        r = cp.Parameter(self.x_dim_linear, value=np.random.randn(self.x_dim_linear))
        w = cp.Parameter(self.x_dim_linear, value=np.random.randn(self.x_dim_linear))
        b = cp.Parameter(1, value=np.random.randn(1))
        nonlinear_score = cp.Parameter(1, value=np.random.randn(1))
        f_der = cp.Parameter(self.x_dim_linear, value=np.random.randn(self.x_dim_linear))

        CCP_target_grad = self.CCP_target_not_batched(x, r, w, b, nonlinear_score, f_der, self.train_slope,
                                                      self.cost_const_kwargs)
        CCP_objective_grad = cp.Maximize(CCP_target_grad)
        CCP_constraints_grad = [x >= self.x_lower_bound, x <= self.x_upper_bound]
        CCP_problem_grad = cp.Problem(CCP_objective_grad, CCP_constraints_grad)
        self.CCP_layer = CvxpyLayer(CCP_problem_grad, parameters=[r, w, b, nonlinear_score, f_der], variables=[x])

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

        X_np = X.numpy()
        w_np = w.detach().numpy()
        b_np = b.detach().numpy()
        nonlinear_score_np = nonlinear_score.detach().numpy()

        self.R.value = X_np
        self.w.value = w_np
        self.b.value = b_np
        self.nonlinear_score.value = nonlinear_score_np

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
        return self.CCP_layer(X, w, b, nonlinear_score_batched, f_der)[0]

    def optimize_X(self, X: tensor, w: tensor, b: tensor, nonlinear_score: tensor, requires_grad: bool = True,
                   **kwargs) -> tensor:
        """
        Given the users' features X, and the current model's parameters w, b, returns the optimal X to which the users
        should move.
        This function approximates the optimal X using the Convex-Concave Procedure (CCP) on the non-differential
        problem, and if needed it solves the differentiable problem to track gradients w.r.t w,b.
        :param X: The user's features, of size (any_batch_size, self.x_dim)
        :param w: The model's weights, of size (self.x_dim).
        :param b: The model's bias, of size (1).
        :param nonlinear_score: The nonlinear score of the user's features, of size (any_batch_size).
        :param requires_grad: Whether to pass the optimal X through the differential problem or not.
        :param kwargs: Any kwargs for the optimization problems.
        :return: The optimal features to which the users should move, with tracked gradients w.r.t w, b.
        """
        slope = self.train_slope if requires_grad else self.eval_slope
        kwargs["slope"] = slope

        # Note that if the first dimension of X is not self.batch_size, we need to split it to chunks of first dimension
        # self.batch_size (and perhaps pad the last batch with zeros to make its first dimension self.batch_size).
        X_batches = self.split_and_pad(X)
        nonlinear_score_batches = self.split_and_pad(nonlinear_score)
        X_opt_batches = [self.solve_optimization_problem(X_batches[i], w, b, nonlinear_score_batches[i], **kwargs)
                         for i in range(len(X_batches))]
        if requires_grad:
            X_opt_batches = [self.solve_differential_optimization_problem(X_batches[i], w, b, nonlinear_score_batches[i],
                                                                          X_opt_batches[i], **kwargs)
                             for i in range(len(X_batches))]

        # Note that if we padded the last batch with zeros, we do not want to keep it after the optimization, hence we
        # slice the result of the following vstack to just the first len(X) rows.
        X_opt = torch.vstack(X_opt_batches)[:len(X)]

        return X_opt

    def CCP_target_batched(self, X, R, w, b, nonlinear_score, f_der, slope, cost_kwargs):
        """
        The CCP target for batched user data.
        :param X: The users' response to the current model, the optimization variable.
        :param R: The users' current features.
        :param w: The model's weights.
        :param b: The model's bias.
        :param nonlinear_score: The nonlinear score of X.
        :param f_der: The derivative of f at the approximation point.
        :param slope: The slope of the CCP approximation.
        :param cost_kwargs: Constant keyword arguments which should be passed to the cost function on computation.
        :return: The target of the batched optimization problem.
        """
        gain = cp.diag(X @ f_der.T) - Commons.g_batch(X, w, b, nonlinear_score, slope)
        cost = self.cost_batched(X, R, **cost_kwargs)
        return gain - cost

    def CCP_target_not_batched(self, x, r, w, b, nonlinear_score, f_der, slope, cost_kwargs):
        """
        The CCP target for non-batched user data.
        :param x: The user's response to the current model, the optimization variable.
        :param r: The user's current features.
        :param w: The model's weights.
        :param b: The model's bias.
        :param nonlinear_score: The nonlinear score of X.
        :param f_der: The derivative of f at the approximation point.
        :param slope: The slope of the CCP approximation.
        :param cost_kwargs: Constant keyword arguments which should be passed to the cost function on computation.
        :return: The target of the non-batched optimization problem.
        """
        gain = x @ f_der - Commons.g(x, w, b, nonlinear_score, slope)
        cost = self.cost_not_batched(x, r, **cost_kwargs)
        return gain - cost

    def split_and_pad(self, A: tensor) -> tensor:
        """
        Splits the given matrix to batches of size self.batch_size, optionally padding the last batch with zeros to get
        its batch dimension to be self.batch_size.
        :param A: The matrix to split to batches.
        :return: The matrix X, split to batches of size batch_size.
        """
        A_batches = list(A.split(self.batch_size))
        A_batches[-1] = self.pad_to_batch_size(A_batches[-1])
        return A_batches

    def pad_to_batch_size(self, A: tensor) -> tensor:
        """
        Pads the given matrix to get its first dimension to become self.batch_size.
        Note: If the first dimension of the matrix is larger than self.batch_size, we raise a ValueError.
        :param A: The matrix to pad to self.batch_size.
        :return: The padded matrix.
        """
        if len(A) == self.batch_size:
            return A
        elif len(A) > self.batch_size:
            error_message = f"The passed matrix is of size {A.shape}, while trying to pad to {self.batch_size}."
            raise ValueError(error_message)

        pad_amount = self.batch_size - len(A)
        if len(A.shape) == 1:
            return F.pad(A, (0, pad_amount), "constant", 0)
        else:
            return F.pad(A, (0, 0, 0, pad_amount), "constant", 0)
