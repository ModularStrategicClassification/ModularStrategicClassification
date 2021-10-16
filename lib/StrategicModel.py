import math
import os
import time
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import warnings

import cvxpy as cp
import numpy as np
from pandas import DataFrame
import torch
from torch import tensor
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, TensorDataset

import lib.CommonDefinitions as Commons
from lib.ResponseMapping import ResponseMapping
import lib.SocialMeasures


class StrategicModel(Module):
    """
    The strategic model class, which learns a linear classifier that takes into account _gaming_ of users.
    Users change their features to maximize their utility, which is defined to be gain - cost.
    """

    def __init__(self, x_dim: int, batch_size: int, loss_fn: Callable = Commons.hinge_loss,
                 cost_fn: Union[str, Callable] = "quad", cost_fn_not_batched: Optional[Callable] = None,
                 cost_fn_torch: Optional[Callable] = None, cost_const_kwargs: Dict = None,
                 utility_reg: Optional[float] = None, burden_reg: Optional[float] = None, recourse_reg: Optional[float] = None,
                 social_measure_dict: Dict = None, train_slope: float = Commons.TRAIN_SLOPE,
                 eval_slope: float = Commons.EVAL_SLOPE, x_lower_bound: float = Commons.X_LOWER_BOUND,
                 x_upper_bound: float = Commons.X_UPPER_BOUND, diff_threshold: float = Commons.DIFF_THRESHOLD,
                 iteration_cap: int = Commons.ITERATION_CAP, strategic: bool = True):
        """
        Initializes a strategic model.
        :param x_dim: The amount of features each user has.
        :param batch_size: The batch size, the amount of users the model will be trained on each batch.
        :param loss_fn: The loss function.
        :param cost_fn: The gaming cost function (for batched user data, cvxpy version). One can supply a string to use one of
            the recognized cost functions, see CommonDefinitions.recognized_cost_functions.
        :param cost_fn_not_batched: The gaming cost function (for non-batched user data, cvxpy version).
        :param cost_fn_torch: The gaming cost function (for batched user data, torch version).
        :param cost_const_kwargs: Constant keyword arguments which should be passed to the cost function on computation.
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
        super().__init__()

        self.x_dim = x_dim
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.strategic = strategic

        # If the user does not supply the model with const keyword arguments for the cost function, we set it to be an empty
        # dictionary (So that the default value of cost_const_kwargs is immutable).
        if cost_const_kwargs is None:
            cost_const_kwargs = {}
        self.cost_const_kwargs = cost_const_kwargs

        # If the user passes a string for the cost function, we check the recognized cost functions dictionary for it.
        if isinstance(cost_fn, str):
            if cost_fn not in Commons.recognized_cost_functions:
                error_message = f"The passed cost function is not recognized by the strategic model, recognized functions " \
                                f"are {Commons.recognized_cost_functions.keys()}."
                raise ValueError(error_message)

            if cost_fn_not_batched is not None:
                warning = "User supplied the strategic model with both a string cost and a non-batched cost function, " \
                          "ignoring the supplied non-batched cost function."
                warnings.warn(warning)
            if cost_fn_torch is not None:
                warning = "User supplied the strategic model with both a string cost and a torch cost function, " \
                          "ignoring the supplied torch cost function."
                warnings.warn(warning)

            cost_fn_dict = Commons.recognized_cost_functions[cost_fn]
            missing_kwargs = [kwarg for kwarg in cost_fn_dict.get("required_kwargs", []) if kwarg not in cost_const_kwargs]
            if len(missing_kwargs) > 0:
                error_message = f"The {cost_fn} cost requires the following const kwargs: {cost_fn_dict['required_kwargs']}," \
                                f"and the following kwargs are missing: {missing_kwargs}. Supply the strategic model with " \
                                f"them via `cost_const_kwargs`."
                raise ValueError(error_message)

            cost_fn = cost_fn_dict.get("cvxpy_batched")
            cost_fn_not_batched = cost_fn_dict.get("cvxpy_not_batched")
            cost_fn_torch = cost_fn_dict.get("torch")

        # The model's parameters
        self.w = Parameter(math.sqrt(1 / x_dim) * (1 - 2 * torch.rand(x_dim, dtype=torch.float64, requires_grad=True)))
        self.b = Parameter(1 - 2 * torch.rand(1, dtype=torch.float64, requires_grad=True))

        # If the user does not supply a non-batched cost function, we wrap the batched cost function to create it.
        if cost_fn_not_batched is None:
            def cost_no_batch(x, r, **kwargs):
                return cost_fn(cp.reshape(x, (1, x_dim)), cp.reshape(r, (1, x_dim)), **kwargs)

            cost_fn_not_batched = cost_no_batch

        # Create the response mapping.
        self.response_mapping = ResponseMapping(x_dim, batch_size, cost_fn, cost_fn_not_batched, cost_const_kwargs,
                                                train_slope, eval_slope, x_lower_bound, x_upper_bound, diff_threshold,
                                                iteration_cap)

        # Social measures (Utility, Burden, Recourse, etc.)
        if social_measure_dict is None:
            social_measure_dict = {}
        if utility_reg is not None:
            if "utility" in social_measure_dict:
                warning = "User supplied both utility_reg and a utility social measure in social_measure_dict, " \
                          "overriding the supplied utility from the dictionary."
                warnings.warn(warning)
            social_measure_dict["utility"] = SocialMeasures.Utility(utility_reg, cost_fn_torch, cost_const_kwargs, train_slope,
                                                                    eval_slope)
        if burden_reg is not None:
            if "burden" in social_measure_dict:
                warning = "User supplied both burden_reg and a burden social measure in the social measure dictionary, " \
                          "overriding the supplied burden from the dictionary."
                warnings.warn(warning)
            social_measure_dict["burden"] = SocialMeasures.Burden(burden_reg, x_dim, cost_fn_not_batched, cost_fn_torch,
                                                                  cost_const_kwargs, x_lower_bound, x_upper_bound)
        if recourse_reg is not None:
            if "recourse" in social_measure_dict:
                warning = "User supplied both recourse_reg and a recourse social measure in the social measure dictionary, " \
                          "overriding the supplied recourse from the dictionary."
                warnings.warn(warning)
            social_measure_dict["recourse"] = SocialMeasures.Recourse(recourse_reg)
        self.social_measure_dict = social_measure_dict

        # Time measurement variables
        self.fit_time = 0
        self.ccp_time = 0

    def forward(self, X: tensor, requires_grad: bool = True) -> Tuple[tensor, tensor]:
        """
        Calculates the score of the given input X, with respect to the current model's parameters (w, b). Also returns the
        optimal X to which the input should move to maximize its utility (returns X if the model is not strategic).
        :param X: The input to calculate its score.
        :param requires_grad: Whether the result should track gradients w.r.t the model's parameters.
        :return: The score and optimal response of the input.
        """
        if self.strategic:
            init_ccp_time = time.time()
            X_opt = self.optimize_X(X, requires_grad=requires_grad)
            self.ccp_time += time.time() - init_ccp_time
        else:
            X_opt = X
        Y_pred = self.score(X_opt)
        return X_opt, Y_pred

    def evaluate(self, X: tensor, Y: tensor, strategic_data: bool = True) -> float:
        """
        Given a pair (X, Y) of features and their corresponding classes, returns the accuracy of the strategic model's
        classification on X.
        :param X: The users' features.
        :param Y: The users' classes
        :param strategic_data: Whether to speculate gaming or not (practically, whether to optimize X or not).
        :return: The accuracy of the strategic model's classification on X.
        """
        if strategic_data:
            X_opt = self.optimize_X(X, requires_grad=False)
        else:
            X_opt = X
        Y_pred = self.score(X_opt)
        return self.calc_accuracy(Y, Y_pred)

    def optimize_X(self, X: tensor, requires_grad: bool = True) -> tensor:
        """
        Given the users' features X, returns the optimal X to which the users should move. Directly calls the response mapping's
        optimize_X method.
        :param X: The users' features.
        :param requires_grad: Whether the results should track gradients w.r.t the model's parameters.
        :return: The optimal response for X w.r.t the current model parameters.
        """
        return self.response_mapping.optimize_X(X, self.w, self.b, requires_grad=requires_grad)

    def score(self, X: tensor) -> tensor:
        """
        Computes the score of the given input.
        :param X: The input to calculate its score.
        :return: The score of X.
        """
        return Commons.score(X, self.w, self.b)

    def loss(self, Y: tensor, Y_pred: tensor, X: tensor, X_opt: tensor) -> tensor:
        """
        Calculates the loss of the model's prediction.
        :param Y: The real classification of the input.
        :param Y_pred: The model's classification of the input.
        :param X: The input.
        :param X_opt: The response of the input, as computed in the forward step.
        :return: The loss of the model's classification.
        """
        loss_val = self.loss_fn(Y, Y_pred)
        for _, social_measure in self.social_measure_dict.items():
            loss_val += social_measure.get_loss_term(Y, Y_pred, X, X_opt, self.w, self.b)
        return loss_val

    def fit(self, X: tensor, Y: tensor, Xval: Optional[tensor] = None, Yval: Optional[tensor] = None,
            opt_class: Type[Optimizer] = Adam, opt_kwargs: Dict = None, shuffle: bool = False, epochs: int = 100,
            epochs_without_improvement_cap: int = 4, verbose: Optional[str] = "batches", save_progress: bool = False,
            path: Optional[str] = None, model_name: Optional[str] = None) -> None:
        """
        Trains the model on the input.
        :param X: The training set.
        :param Y: The training classes.
        :param Xval: The validation set.
        :param Yval: The validation labels.
        :param opt_class: The class of the optimizer used in the training.
        :param opt_kwargs: Any keyword arguments to be passed to the optimizer.
        :param shuffle: Whether to shuffle the training and validation sets in the training.
        :param epochs: The amount of epochs to train the model.
        :param epochs_without_improvement_cap: The cap on the amount of epochs in which the model doesn't improve (i.e. doesn't
            increase its accuracy).
        :param verbose: Whether to write messages during the training process. Set to "batches" to write at the end of every
            batch, "epochs" to write at the end of every epoch, or None to only write at the end of the training process.
        :param save_progress: Whether to save the progress of the model during training. One must supply a path and model_name
            to save the model's progress.
        :param path: A path to a directory in which the model will be saved. Can be None, in this case, the model will not be
            saved to a file.
        :param model_name: The model's name, used for saving.
        """
        # This initialization makes opt_kwargs' default value immutable.
        if opt_kwargs is None:
            opt_kwargs = {"lr": 5e-1}
        optimizer = opt_class(self.parameters(), **opt_kwargs)

        train_dataset = TensorDataset(X, Y)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=shuffle)
        train_batches = len(train_dataloader)
        train_losses = []
        train_errors = []

        for _, social_measure in self.social_measure_dict.items():
            social_measure.init_fit_train(epochs, verbose)

        validation = (Xval is not None)
        if validation:
            assert (Yval is not None)
            validation_dataset = TensorDataset(Xval, Yval)
            validation_dataloader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=shuffle)
            validation_losses = []
            validation_errors = []
            best_validation_error = 1
            consecutive_no_improvement = 0

            for _, social_measure in self.social_measure_dict.items():
                social_measure.init_fit_validation(verbose)
        else:
            validation_losses = None
            validation_errors = None

        if save_progress:
            self.save_initial_data(path, model_name)
        best_model_state_dict = self.state_dict()

        print_every_epoch = verbose in ["batches", "epochs"]
        print_every_batch = verbose in ["batches"]

        init_fit_time = time.time()
        for epoch in range(epochs):
            if print_every_epoch:
                print(f"Starting epoch {epoch + 1:03d} / {epochs:03d}.")

            init_epoch_time = time.time()
            batch = 0
            train_losses.append([])
            train_errors.append([])

            for _, social_measure in self.social_measure_dict.items():
                social_measure.begin_epoch(epoch, validation)

            for Xbatch, Ybatch in train_dataloader:
                for _, social_measure in self.social_measure_dict.items():
                    social_measure.begin_batch(epoch, batch, Xbatch, Ybatch)

                optimizer.zero_grad()
                Xbatch_opt, Ybatch_pred = self.forward(Xbatch, requires_grad=True)
                loss = self.loss(Ybatch, Ybatch_pred, Xbatch, Xbatch_opt)
                loss.backward()
                optimizer.step()

                train_losses[epoch].append(loss.item())
                accuracy = self.calc_accuracy(Ybatch, Ybatch_pred)
                train_errors[epoch].append(1 - accuracy)

                for _, social_measure in self.social_measure_dict.items():
                    social_measure.end_batch(epoch, batch, Xbatch, Ybatch, Xbatch_opt, Ybatch_pred, self.w, self.b)

                if print_every_batch:
                    self.print_batch_info(epoch, epochs, batch, train_batches, train_losses, train_errors)

                if save_progress:
                    self.save_batch_progress(epoch, batch, Xbatch, Xbatch_opt, Ybatch, Ybatch_pred, path, model_name)

                batch += 1

            if validation:
                if print_every_batch:
                    print("  Finished training step, calculating validation loss and accuracy.")

                batch = 0

                validation_losses.append([])
                validation_errors.append([])

                for _, social_measure in self.social_measure_dict.items():
                    social_measure.begin_validation(epoch)

                with torch.no_grad():
                    for Xbatch, Ybatch in validation_dataloader:
                        for _, social_measure in self.social_measure_dict.items():
                            social_measure.begin_validation_batch(epoch, batch, Xbatch, Ybatch)

                        Xbatch_opt, Ybatch_pred = self.forward(Xbatch, requires_grad=False)
                        loss = self.loss(Ybatch, Ybatch_pred, Xbatch, Xbatch_opt)
                        validation_losses[epoch].append(loss.item())
                        accuracy = self.calc_accuracy(Ybatch, Ybatch_pred)
                        validation_errors[epoch].append(1 - accuracy)

                        for _, social_measure in self.social_measure_dict.items():
                            social_measure.end_validation_batch(epoch, batch, Xbatch, Ybatch, Xbatch_opt, Ybatch_pred, self.w,
                                                                self.b)

                    batch += 1

            final_epoch_time = time.time()
            epoch_time = final_epoch_time - init_epoch_time
            if print_every_epoch:
                self.print_epoch_info(epoch, epochs, epoch_time, validation_losses, validation_errors)

            for _, social_measure in self.social_measure_dict.items():
                social_measure.end_epoch(epoch)

            if validation:
                avg_validation_error = np.mean(validation_errors[epoch])
                if avg_validation_error < best_validation_error:
                    best_validation_error = avg_validation_error
                    consecutive_no_improvement = 0
                    if epoch != 0 and print_every_epoch:
                        print("Validation accuracy improved.")
                    self.save_model(path, model_name, print_every_epoch)
                    best_model_state_dict = self.state_dict()
                else:
                    consecutive_no_improvement += 1
                    if consecutive_no_improvement >= epochs_without_improvement_cap and epoch < epochs - 1:
                        if print_every_epoch:
                            print(f"Ending training due to {consecutive_no_improvement} consecutive epochs without "
                                  f"improvement in validation accuracy.")
                        break
            else:
                self.save_model(path, model_name, print_every_epoch)
                best_model_state_dict = self.state_dict()

        final_fit_time = time.time()
        self.fit_time = final_fit_time - init_fit_time
        if self.fit_time <= 60:
            print(f"Total training time: {self.fit_time} seconds.")
        else:
            print(f"Total training time: {self.fit_time / 60} minutes ({self.fit_time} seconds).")

        # Try and load the model from a file. If the loading fails (if there is not path/model_name), only loads the best
        # model's state dict (social measures' load is not called).
        if not self.load_model(path, model_name, print_every_epoch):
            self.load_state_dict(best_model_state_dict)

    def normalize_parameters(self) -> None:
        """
        Normalize the model's parameters (w, b).
        """
        with torch.no_grad():
            norm = torch.sqrt(torch.sum(self.w ** 2) + self.b ** 2)
            self.w /= norm
            self.b /= norm

    def save_model(self, path: Optional[str], model_name: Optional[str], print_message: bool = True) -> None:
        """
        Saves the model to the supplied path with the given name. If the path or name are not supplied, nothing happens.
        :param path: The directory in which the model will be saved.
        :param model_name: The model's name.
        :param print_message: Whether to print a message that the model was saved.
        """
        if path is None or model_name is None:
            return

        if not os.path.exists(path):
            os.makedirs(path)

        filename = self.get_filename(path, model_name, "model", "pt")
        torch.save(self.state_dict(), filename)
        for _, social_measure in self.social_measure_dict.items():
            social_measure.save(path, model_name, print_message)

        if print_message:
            print(f"Model saved to {filename}.")

    def load_model(self, path: str, model_name: str, print_message: bool = True) -> bool:
        """
        Loads the model from the specified path with the given name.
        :param path: The directory from which the model will be loaded.
        :param model_name: The model's name.
        :param print_message: Whether to print a message that the model was loaded.
        :return: Whether the model was loaded.
        """
        if path is None or model_name is None:
            return False

        filename = self.get_filename(path, model_name, "model", "pt")
        if not os.path.isfile(filename):
            return False

        self.load_state_dict(torch.load(filename))
        for _, social_measure in self.social_measure_dict.items():
            social_measure.load(path, model_name, print_message)

        if print_message:
            print(f"Model loaded from {filename}.")

        return True

    def print_batch_info(self, epoch: int, epochs: int, batch: int, batches: int, train_losses: List[List[float]],
                         train_errors: List[List[float]]) -> None:
        """
        Prints info about the train batch after it is done.
        :param epoch: The epoch number (starting at 0).
        :param epochs: The number of epochs.
        :param batch: The batch number (starting at 0).
        :param batches: The number of batches.
        :param train_losses: The train losses up to this point.
        :param train_errors: The train errors up to this point.
        """
        loss = train_losses[epoch][batch]
        error = train_errors[epoch][batch]

        print(f"  batch {batch + 1:03d} / {batches:03d} | loss: {loss:3.5f} | error: {error:3.5f}", end="")
        for name, social_measure in self.social_measure_dict.items():
            sm_string = social_measure.get_batch_info(epoch, batch)
            if len(sm_string) > 0:
                print(f" | {name}: {sm_string}", end="")
        print("")

    def print_epoch_info(self, epoch: int, epochs: int, epoch_time: float, validation_losses: Optional[List[List[float]]],
                         validation_errors: Optional[List[List[float]]]) -> None:
        """
        Prints info about the epoch after it is done.
        :param epoch: The epoch number (starting at 0).
        :param epochs: The number of epochs.
        :param epoch_time: The time the epoch took.
        :param validation_losses: The validation losses up to this point.
        :param validation_errors: The validation errors up to this point.
        """
        print(f"epoch {epoch + 1:03d} / {epochs:03d} | time: {round(epoch_time):03d} sec", end="")
        if validation_losses is not None:
            loss = np.mean(validation_losses[epoch])
            error = np.mean(validation_errors[epoch])
            print(f" | loss: {loss:3.5f} | error: {error:3.5f}", end="")
            for name, social_measure in self.social_measure_dict.items():
                sm_string = social_measure.get_validation_info(epoch)
                if len(sm_string) > 0:
                    print(f" | {name}: {sm_string}", end="")
        print("")

    def save_initial_data(self, path: str, model_name: str) -> None:
        """
        Saves the initial data of the model.
        If path or model_name is None, raises a ValueError (one must supply the path and model name in order to save data).
        :param path: The directory in which we save the data.
        :param model_name: The model's name.
        """
        if path is None or model_name is None:
            error_message = "In order to save the model's progression, one must supply a path and model name."
            raise ValueError(error_message)

        filename = self.get_filename(path, model_name, "progress", "csv")
        init_data = self.get_batch_data_for_saving(-1, -1, X=torch.zeros((1, self.x_dim)), X_opt=torch.zeros((1, self.x_dim)),
                                                   Y=torch.zeros(1), Y_pred=torch.zeros(1))
        DataFrame(init_data).to_csv(filename, index=False)

    def save_batch_progress(self, epoch: int, batch: int, X: tensor, X_opt: tensor, Y: tensor, Y_pred: tensor, path: str,
                            model_name: str) -> None:
        """
        Saves the progress of the given batch into the already created csv data file.
        :param epoch: The epoch number.
        :param batch: The batch number.
        :param X: The users' features.
        :param X_opt: The users' responses.
        :param Y: The users' ground-truth classes.
        :param Y_pred: The model's prediction of the users' classes.
        :param path: The directory in which we save the data.
        :param model_name: The model's name.
        """
        filename = self.get_filename(path, model_name, "progress", "csv")
        epoch_data = self.get_batch_data_for_saving(epoch + 1, batch + 1, X, X_opt, Y, Y_pred)
        DataFrame(epoch_data).to_csv(filename, mode="a", header=False, index=False)

    def get_batch_data_for_saving(self, epoch: int, batch: int, X: tensor, X_opt: tensor, Y: tensor, Y_pred: tensor) -> \
            List[Dict[str, float]]:
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
        w_dict = {f"w_{j}": self.w[j].detach().numpy() for j in range(self.linear_x_dim)}

        def x_dict(row):
            return {f"x_{j}": X[row][j].detach().numpy() for j in range(self.x_dim)}

        def x_opt_dict(row):
            return {f"x_opt_{j}": X_opt[row][j].detach().numpy() for j in range(self.x_dim)}

        return [{"epoch": epoch, "batch": batch,
                 **w_dict, "b": self.b[0].detach().numpy(),
                 **x_dict(row), "y": Y[row].detach().numpy(),
                 **x_opt_dict(row), "y_pred": Y_pred[row].detach().numpy()} for row in range(len(X))]

    @staticmethod
    def calc_accuracy(Y: tensor, Y_pred: tensor) -> float:
        """
        Calculates the accuracy of the prediction Y_pred w.r.t the ground-truth classes Y.
        :param Y: The ground-truth classes.
        :param Y_pred: The prediction of the model.
        :return: The accuracy of the prediction.
        """
        with torch.no_grad():
            Y_pred = torch.sign(Y_pred)
            temp = Y - Y_pred
            accuracy = len(temp[temp == 0]) / len(Y)
        return accuracy

    @staticmethod
    def get_filename(path: str, name: str, suffix: str, extension: str) -> str:
        """
        Gets the file name of the file with the given path, name, suffix and extension.
        :param path: The directory of the file.
        :param name: The name of the file.
        :param suffix: The suffix of the file (comes after the name).
        :param extension: The extension of the file.
        :return: The full filename to the file.
        """
        return f"{path}/{name}_{suffix}.{extension}"
