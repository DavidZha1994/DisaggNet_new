#################################################################################################################
#
# @copyright : ©2025 EDF
# @author : Adrien Petralia
# @description : NILMFormer - PyTorch Trainer
#
#################################################################################################################

import os
import time
import logging

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from src.helpers.metrics import NILMmetrics


class SeqToSeqTrainer:
    def __init__(
        self,
        model,
        train_loader,
        valid_loader=None,
        learning_rate=1e-3,
        weight_decay=1e-2,
        criterion=nn.MSELoss(),
        consumption_pred=True,
        patience_es=None,
        patience_rlr=None,
        device="cuda",
        all_gpu=False,
        valid_criterion=None,
        training_in_model=False,
        loss_in_model=False,
        f_metrics=NILMmetrics(),
        n_warmup_epochs=0,
        verbose=True,
        plotloss=True,
        save_fig=False,
        path_fig=None,
        save_checkpoint=False,
        path_checkpoint=None,
    ):
        """
        PyTorch Model Trainer Class for SeqToSeq NILM (per timestamps estimation)

        Can be either: classification, values in [0,1] or energy power estimation for each timesteps
        """

        # =======================class variables======================= #
        self.model = model
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.all_gpu = all_gpu
        self.verbose = verbose
        self.plotloss = plotloss
        self.save_checkpoint = save_checkpoint
        self.path_checkpoint = path_checkpoint
        self.save_fig = save_fig
        self.path_fig = path_fig
        self.patience_rlr = patience_rlr
        self.patience_es = patience_es
        self.n_warmup_epochs = n_warmup_epochs
        self.consumption_pred = consumption_pred
        self.f_metrics = f_metrics
        self.loss_in_model = loss_in_model
        self.training_in_model = training_in_model

        if self.training_in_model:
            assert hasattr(self.model, "train_one_epoch")

        self.train_criterion = criterion
        if valid_criterion is None:
            self.valid_criterion = criterion
        else:
            self.valid_criterion = valid_criterion

        if self.path_checkpoint is not None:
            self.path_checkpoint = path_checkpoint
        else:
            self.path_checkpoint = os.getcwd() + os.sep + "model"

        if self.patience_rlr is not None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                "min",
                patience=self.patience_rlr,
                eps=1e-7,
            )

        self.log = {}
        self.train_time = 0
        self.eval_time = 0
        self.passed_epochs = 0
        self.best_loss = np.inf
        self.loss_train_history = []
        self.loss_valid_history = []
        self.accuracy_train_history = []
        self.accuracy_valid_history = []

        if self.patience_es is not None:
            self.early_stopping = EarlyStopper(patience=self.patience_es)

        if self.all_gpu:
            # =========== Dummy forward to intialize Lazy Module =========== #
            self.model.to("cpu")
            for ts, _, _ in train_loader:
                self.model(torch.rand(ts.shape))
                break
            # =========== Data Parrallel Module call =========== #
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

    def train(self, n_epochs=10):
        """
        Public function : master training loop over epochs
        """

        # flag_es = 0
        tmp_time = time.time()

        for epoch in range(n_epochs):
            # =======================one epoch======================= #
            if self.training_in_model:
                self.model.train()
                if self.all_gpu:
                    train_loss = self.model.module.train_one_epoch(
                        loader=self.train_loader,
                        optimizer=self.optimizer,
                        device=self.device,
                    )
                else:
                    train_loss = self.model.train_one_epoch(
                        loader=self.train_loader,
                        optimizer=self.optimizer,
                        device=self.device,
                    )
            else:
                train_loss = self.__train()
            self.loss_train_history.append(train_loss)
            if self.valid_loader is not None:
                valid_loss = self.__evaluate()
                self.loss_valid_history.append(valid_loss)
            else:
                valid_loss = train_loss

            # =======================reduce lr======================= #
            if self.patience_rlr:
                self.scheduler.step(valid_loss)

            # ===================early stoppping=================== #
            if self.patience_es is not None:
                if (
                    self.passed_epochs > self.n_warmup_epochs
                ):  # Avoid n_warmup_epochs first epochs
                    if self.early_stopping.early_stop(valid_loss):
                        # flag_es  = 1
                        epoch + 1
                        self.passed_epochs += 1
                        if self.verbose:
                            logging.info(
                                "Early stopping after {} epochs !".format(epoch + 1)
                            )
                        break

            # =======================verbose======================= #
            if self.verbose:
                logging.info("Epoch [{}/{}]".format(epoch + 1, n_epochs))
                logging.info("    Train loss : {:.4f}".format(train_loss))

                if self.valid_loader is not None:
                    logging.info("    Valid  loss : {:.4f}".format(valid_loss))

            # =======================save log======================= #
            if (
                valid_loss <= self.best_loss
                and self.passed_epochs >= self.n_warmup_epochs
            ):
                self.best_loss = valid_loss
                self.log = {
                    "model_state_dict": self.model.module.state_dict()
                    if self.device == "cuda" and self.all_gpu
                    else self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss_train_history": self.loss_train_history,
                    "loss_valid_history": self.loss_valid_history,
                    "value_best_loss": self.best_loss,
                    "epoch_best_loss": self.passed_epochs,
                    "time_best_loss": round((time.time() - tmp_time), 3),
                }
                if self.save_checkpoint:
                    self.save()

            self.passed_epochs += 1

        self.train_time = round((time.time() - tmp_time), 3)

        if self.plotloss:
            self.plot_history()

        if self.save_checkpoint:
            self.log["best_model_state_dict"] = torch.load(
                self.path_checkpoint + ".pt"
            )["model_state_dict"]

        # =======================update log======================= #
        self.log["training_time"] = self.train_time
        self.log["loss_train_history"] = self.loss_train_history
        self.log["loss_valid_history"] = self.loss_valid_history

        if self.save_checkpoint:
            self.save()
        return

    def evaluate(
        self,
        test_loader,
        scaler=None,
        factor_scaling=1,
        save_outputs=False,
        mask="test_metrics",
        threshold_small_values=0,
        threshold_activation=None,
        apply_sigmoid=True,
    ):
        """
        Public function : model evaluation on test dataset
        """
        loss_valid = 0

        y = np.array([])
        y_hat = np.array([])
        y_win = np.array([])
        y_hat_win = np.array([])
        y_state = np.array([])
        y_hat_state = np.array([])

        start_time = time.time()
        with torch.no_grad():
            for ts_agg, appl, state in test_loader:
                self.model.eval()

                # ===================variables=================== #
                ts_agg = torch.Tensor(ts_agg.float()).to(self.device)

                if self.consumption_pred:
                    target = torch.Tensor(appl.float()).to(self.device)
                else:
                    target = torch.Tensor(state.float()).to(self.device)

                # ===================forward and loss===================== #
                if self.loss_in_model:
                    pred, _ = self.model(ts_agg, target)
                else:
                    pred = self.model(ts_agg)

                loss = self.valid_criterion(pred, target)
                loss_valid += loss.item()

                # ===================Evaluate using provided metrics===================== #
                if self.consumption_pred:
                    if scaler is not None:
                        target = scaler.inverse_transform_appliance(target)
                        pred = scaler.inverse_transform_appliance(pred)
                    else:
                        target = target * factor_scaling
                        pred = pred * factor_scaling

                    pred[pred < threshold_small_values] = 0

                    target_win = target.sum(dim=-1)
                    pred_win = pred.sum(dim=-1)

                    y = (
                        np.concatenate(
                            (y, torch.flatten(target).detach().cpu().numpy())
                        )
                        if y.size
                        else torch.flatten(target).detach().cpu().numpy()
                    )
                    y_hat = (
                        np.concatenate(
                            (y_hat, torch.flatten(pred).detach().cpu().numpy())
                        )
                        if y_hat.size
                        else torch.flatten(pred).detach().cpu().numpy()
                    )
                    y_win = (
                        np.concatenate(
                            (y_win, torch.flatten(target_win).detach().cpu().numpy())
                        )
                        if y_win.size
                        else torch.flatten(target_win).detach().cpu().numpy()
                    )
                    y_hat_win = (
                        np.concatenate(
                            (y_hat_win, torch.flatten(pred_win).detach().cpu().numpy())
                        )
                        if y_hat_win.size
                        else torch.flatten(pred_win).detach().cpu().numpy()
                    )
                    y_state = (
                        np.concatenate((y_state, state.flatten()))
                        if y_state.size
                        else state.flatten()
                    )

                else:
                    if apply_sigmoid:
                        pred = nn.Sigmoid()(pred)

                    y_state = (
                        np.concatenate((y_state, state.flatten()))
                        if y_state.size
                        else state.flatten()
                    )
                    y_hat_state = (
                        np.concatenate(
                            (y_hat_state, torch.flatten(pred).detach().cpu().numpy())
                        )
                        if y_hat_state.size
                        else torch.flatten(pred).detach().cpu().numpy()
                    )

        loss_valid = loss_valid / len(self.valid_loader)

        if self.consumption_pred:
            metrics_timestamp = self.f_metrics(
                y,
                y_hat,
                y_state,
                y_hat_state=(
                    y_hat
                    > (
                        threshold_activation
                        if threshold_activation is not None
                        else threshold_small_values
                    )
                ).astype(dtype=int),
            )
            metrics_win = self.f_metrics(y_win, y_hat_win)

            self.log[mask + "_timestamp"] = metrics_timestamp
            self.log[mask + "_win"] = metrics_win
        else:
            metrics = self.f_metrics(
                y=None, y_hat=None, y_state=y_state, y_hat_state=y_state
            )
            self.log[mask + "_timestamp"] = metrics

        self.eval_time = round((time.time() - start_time), 3)

        self.log[mask + "_time"] = self.eval_time

        if save_outputs:
            self.log[mask + "_yhat"] = y_hat

            if y_hat_win.size:
                self.log[mask + "_yhat_win"] = y_hat

        if self.save_checkpoint:
            self.save()

        return np.mean(loss_valid)

    def save(self):
        """
        Public function : save log
        """
        torch.save(self.log, self.path_checkpoint + ".pt")
        return

    def plot_history(self):
        """
        Public function : plot loss history
        """
        plt.figure()
        plt.plot(range(self.passed_epochs), self.loss_train_history, label="Train loss")
        if self.valid_loader is not None:
            plt.plot(
                range(self.passed_epochs), self.loss_valid_history, label="Valid loss"
            )
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.legend()
        if self.path_fig:
            plt.savefig(self.path_fig)
        else:
            plt.show()
        return

    def reduce_lr(self, new_lr):
        """
        Public function : update learning of the optimizer
        """
        for g in self.model.optimizer.param_groups:
            g["lr"] = new_lr

        return

    def restore_best_weights(self):
        """
        Public function : load best model state dict parameters met during training.
        """
        try:
            if self.all_gpu:
                self.model.module.load_state_dict(self.log["best_model_state_dict"])
            else:
                self.model.load_state_dict(self.log["best_model_state_dict"])
            logging.info("Restored best model met during training.")
        except KeyError:
            logging.info("Error during loading log checkpoint state dict : no update.")
        return

    def __train(self):
        """
        Private function : model training loop over data loader
        """
        loss_train = 0

        for ts_agg, appl, states in self.train_loader:
            self.model.train()

            # ===================variables=================== #
            ts_agg = torch.Tensor(ts_agg.float()).to(self.device)
            if self.consumption_pred:
                target = torch.Tensor(appl.float()).to(self.device)
            else:
                target = torch.Tensor(states.float()).to(self.device)

            # ===================forward===================== #
            self.optimizer.zero_grad()

            if self.loss_in_model:
                pred, loss = self.model(ts_agg, target)
            else:
                if self.consumption_pred:
                    pred = self.model(ts_agg)
                else:
                    pred = nn.Sigmoid()(self.model(ts_agg))

                loss = self.train_criterion(pred, target)

            # ===================backward==================== #
            loss_train += loss.item()
            loss.backward()
            self.optimizer.step()

        loss_train = loss_train / len(self.train_loader)

        return loss_train

    def __evaluate(self):
        """
        Private function : model evaluation loop over data loader
        """
        loss_valid = 0

        with torch.no_grad():
            for ts_agg, appl, states in self.valid_loader:
                self.model.eval()

                # ===================variables=================== #

                ts_agg = torch.Tensor(ts_agg.float()).to(self.device)
                if self.consumption_pred:
                    target = torch.Tensor(appl.float()).to(self.device)
                else:
                    target = torch.Tensor(states.float()).to(self.device)

                # ===================forward=================== #
                if self.loss_in_model:
                    pred, loss = self.model(ts_agg, target)
                else:
                    pred = self.model(ts_agg)
                    loss = self.valid_criterion(pred, target)

                loss_valid += loss.item()

        loss_valid = loss_valid / len(self.valid_loader)

        return loss_valid


class TserTrainer:
    def __init__(
        self,
        model,
        train_loader,
        valid_loader=None,
        learning_rate=1e-3,
        weight_decay=1e-2,
        criterion=nn.MSELoss(),
        patience_es=None,
        patience_rlr=None,
        device="cuda",
        all_gpu=False,
        valid_criterion=None,
        training_in_model=False,
        loss_in_model=False,
        f_metrics=NILMmetrics(),
        n_warmup_epochs=0,
        verbose=True,
        plotloss=True,
        save_fig=False,
        path_fig=None,
        save_checkpoint=False,
        path_checkpoint=None,
    ):
        """
        PyTorch Model Trainer Class for Time Series Extrinsic Regression for NILM
        """

        # =======================class variables======================= #
        self.model = model
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.all_gpu = all_gpu
        self.verbose = verbose
        self.plotloss = plotloss
        self.save_checkpoint = save_checkpoint
        self.path_checkpoint = path_checkpoint
        self.save_fig = save_fig
        self.path_fig = path_fig
        self.patience_rlr = patience_rlr
        self.patience_es = patience_es
        self.n_warmup_epochs = n_warmup_epochs
        self.f_metrics = f_metrics
        self.loss_in_model = loss_in_model
        self.training_in_model = training_in_model

        if self.training_in_model:
            assert hasattr(self.model, "train_one_epoch")

        self.train_criterion = criterion
        if valid_criterion is None:
            self.valid_criterion = criterion
        else:
            self.valid_criterion = valid_criterion

        if self.path_checkpoint is not None:
            self.path_checkpoint = path_checkpoint
        else:
            self.path_checkpoint = os.getcwd() + os.sep + "model"

        if self.patience_rlr is not None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                "min",
                patience=self.patience_rlr,
                eps=1e-7,
            )

        self.log = {}
        self.train_time = 0
        self.eval_time = 0
        self.passed_epochs = 0
        self.best_loss = np.inf
        self.loss_train_history = []
        self.loss_valid_history = []
        self.accuracy_train_history = []
        self.accuracy_valid_history = []

        if self.patience_es is not None:
            self.early_stopping = EarlyStopper(patience=self.patience_es)

        if self.all_gpu:
            # =========== Dummy forward to intialize Lazy Module =========== #
            self.model.to("cpu")
            for ts, _ in train_loader:
                self.model(torch.rand(ts.shape))
                break
            # =========== Data Parrallel Module call =========== #
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

    def train(self, n_epochs=10):
        """
        Public function : master training loop over epochs
        """

        tmp_time = time.time()

        for epoch in range(n_epochs):
            # =======================one epoch======================= #
            if self.training_in_model:
                self.model.train()
                if self.all_gpu:
                    train_loss = self.model.module.train_one_epoch(
                        loader=self.train_loader,
                        optimizer=self.optimizer,
                        device=self.device,
                    )
                else:
                    train_loss = self.model.train_one_epoch(
                        loader=self.train_loader,
                        optimizer=self.optimizer,
                        device=self.device,
                    )
            else:
                train_loss = self.__train()
            self.loss_train_history.append(train_loss)
            if self.valid_loader is not None:
                valid_loss = self.__evaluate()
                self.loss_valid_history.append(valid_loss)
            else:
                valid_loss = train_loss

            # =======================reduce lr======================= #
            if self.patience_rlr:
                self.scheduler.step(valid_loss)

            # ===================early stoppping=================== #
            if self.patience_es is not None:
                if (
                    self.passed_epochs > self.n_warmup_epochs
                ):  # Avoid n_warmup_epochs first epochs
                    if self.early_stopping.early_stop(valid_loss):
                        self.passed_epochs += 1
                        if self.verbose:
                            logging.info(
                                "Early stopping after {} epochs !".format(epoch + 1)
                            )
                        break

            # =======================verbose======================= #
            if self.verbose:
                logging.info("Epoch [{}/{}]".format(epoch + 1, n_epochs))
                logging.info("    Train loss : {:.4f}".format(train_loss))

                if self.valid_loader is not None:
                    logging.info("    Valid  loss : {:.4f}".format(valid_loss))

            # =======================save log======================= #
            if (
                valid_loss <= self.best_loss
                and self.passed_epochs >= self.n_warmup_epochs
            ):
                self.best_loss = valid_loss
                self.log = {
                    "model_state_dict": self.model.module.state_dict()
                    if self.device == "cuda" and self.all_gpu
                    else self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss_train_history": self.loss_train_history,
                    "loss_valid_history": self.loss_valid_history,
                    "value_best_loss": self.best_loss,
                    "epoch_best_loss": self.passed_epochs,
                    "time_best_loss": round((time.time() - tmp_time), 3),
                }
                if self.save_checkpoint:
                    self.save()

            self.passed_epochs += 1

        self.train_time = round((time.time() - tmp_time), 3)

        if self.plotloss:
            self.plot_history()

        if self.save_checkpoint:
            self.log["best_model_state_dict"] = torch.load(
                self.path_checkpoint + ".pt"
            )["model_state_dict"]

        # =======================update log======================= #
        self.log["training_time"] = self.train_time
        self.log["loss_train_history"] = self.loss_train_history
        self.log["loss_valid_history"] = self.loss_valid_history

        if self.save_checkpoint:
            self.save()
        return

    def evaluate(
        self,
        test_loader,
        scaler=None,
        factor_scaling=1,
        save_outputs=False,
        mask="test_metrics",
        threshold_small_values=0,
    ):
        """
        Public function : model evaluation on test dataset
        """
        loss_valid = 0

        y = np.array([])
        y_hat = np.array([])

        start_time = time.time()
        with torch.no_grad():
            for ts_agg, target in test_loader:
                self.model.eval()

                # ===================variables=================== #
                ts_agg = torch.Tensor(ts_agg.float()).to(self.device)
                target = torch.Tensor(target.float()).to(self.device)

                if len(target.shape) == 1:
                    target = target.unsqueeze(1)

                # ===================forward and loss===================== #
                if self.loss_in_model:
                    pred, _ = self.model(ts_agg, target)
                else:
                    pred = self.model(ts_agg)

                loss = self.valid_criterion(pred, target)
                loss_valid += loss.item()

                # ===================Evaluate using provided metrics===================== #
                if scaler is not None:
                    target = scaler.inverse_transform_appliance(target)
                    pred = scaler.inverse_transform_appliance(pred)
                else:
                    target = target * factor_scaling
                    pred = pred * factor_scaling

                pred[pred < threshold_small_values] = 0

                y = (
                    np.concatenate((y, torch.flatten(target).detach().cpu().numpy()))
                    if y.size
                    else torch.flatten(target).detach().cpu().numpy()
                )
                y_hat = (
                    np.concatenate((y_hat, torch.flatten(pred).detach().cpu().numpy()))
                    if y_hat.size
                    else torch.flatten(pred).detach().cpu().numpy()
                )

        loss_valid = loss_valid / len(self.valid_loader)

        metrics_win = self.f_metrics(y, y_hat)
        self.log[mask + "_win"] = metrics_win

        self.eval_time = round((time.time() - start_time), 3)

        self.log[mask + "_time"] = self.eval_time

        if save_outputs:
            self.log[mask + "_yhat_win"] = y_hat

        if self.save_checkpoint:
            self.save()

        return np.mean(loss_valid)

    def save(self):
        """
        Public function : save log
        """
        torch.save(self.log, self.path_checkpoint + ".pt")
        return

    def plot_history(self):
        """
        Public function : plot loss history
        """
        plt.figure()
        plt.plot(range(self.passed_epochs), self.loss_train_history, label="Train loss")
        if self.valid_loader is not None:
            plt.plot(
                range(self.passed_epochs), self.loss_valid_history, label="Valid loss"
            )
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.legend()
        if self.path_fig:
            plt.savefig(self.path_fig)
        else:
            plt.show()
        return

    def reduce_lr(self, new_lr):
        """
        Public function : update learning of the optimizer
        """
        for g in self.model.optimizer.param_groups:
            g["lr"] = new_lr

        return

    def restore_best_weights(self):
        """
        Public function : load best model state dict parameters met during training.
        """
        try:
            if self.all_gpu:
                self.model.module.load_state_dict(self.log["best_model_state_dict"])
            else:
                self.model.load_state_dict(self.log["best_model_state_dict"])
            logging.info("Restored best model met during training.")
        except KeyError:
            logging.info("Error during loading log checkpoint state dict : no update.")
        return

    def __train(self):
        """
        Private function : model training loop over data loader
        """
        loss_train = 0

        for ts_agg, target in self.train_loader:
            self.model.train()

            # ===================variables=================== #
            ts_agg = torch.Tensor(ts_agg.float()).to(self.device)
            target = torch.Tensor(target.float()).to(self.device)

            if len(target.shape) == 1:
                target = target.unsqueeze(1)

            # ===================forward===================== #
            self.optimizer.zero_grad()

            if self.loss_in_model:
                pred, loss = self.model(ts_agg, target)
            else:
                pred = self.model(ts_agg)

                loss = self.train_criterion(pred, target)

            # ===================backward==================== #
            loss_train += loss.item()
            loss.backward()
            self.optimizer.step()

        loss_train = loss_train / len(self.train_loader)

        return loss_train

    def __evaluate(self):
        """
        Private function : model evaluation loop over data loader
        """
        loss_valid = 0

        with torch.no_grad():
            for ts_agg, target in self.valid_loader:
                self.model.eval()

                # ===================variables=================== #
                ts_agg = torch.Tensor(ts_agg.float(), device=self.device)
                target = torch.Tensor(target.float(), device=self.device)

                if len(target.shape) == 1:
                    target = target.unsqueeze(1)

                # ===================forward=================== #
                if self.loss_in_model:
                    pred, loss = self.model(ts_agg, target)
                else:
                    pred = self.model(ts_agg)
                    loss = self.valid_criterion(pred, target)

                loss_valid += loss.item()

        loss_valid = loss_valid / len(self.valid_loader)

        return loss_valid


class BasedSelfPretrainer(object):
    def __init__(
        self,
        model,
        train_loader,
        valid_loader=None,
        learning_rate=1e-3,
        weight_decay=0,
        name_scheduler=None,
        dict_params_scheduler=None,
        warmup_duration=None,
        criterion=nn.MSELoss(),
        mask=None,
        loss_in_model=False,
        device="cuda",
        all_gpu=False,
        verbose=True,
        plotloss=True,
        save_fig=False,
        path_fig=None,
        save_only_core=False,
        save_checkpoint=False,
        path_checkpoint=None,
    ):
        # =======================class variables======================= #
        self.device = device
        self.all_gpu = all_gpu
        self.model = model
        self.criterion = criterion
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.mask = mask
        self.verbose = verbose
        self.plotloss = plotloss
        self.save_checkpoint = save_checkpoint
        self.path_checkpoint = path_checkpoint
        self.save_fig = save_fig
        self.path_fig = path_fig
        self.save_only_core = save_only_core
        self.loss_in_model = loss_in_model
        self.name_scheduler = name_scheduler

        if name_scheduler is None:
            self.scheduler = None
        else:
            assert isinstance(dict_params_scheduler, dict)

            if name_scheduler == "MultiStepLR":
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer,
                    milestones=dict_params_scheduler["milestones"],
                    gamma=dict_params_scheduler["gamma"],
                    verbose=self.verbose,
                )

            elif name_scheduler == "CosineAnnealingLR":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=dict_params_scheduler["T_max"],
                    eta_min=dict_params_scheduler["eta_min"],
                    verbose=self.verbose,
                )

            elif name_scheduler == "CosineAnnealingWarmRestarts":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=dict_params_scheduler["T_0"],
                    T_mult=dict_params_scheduler["T_mult"],
                    eta_min=dict_params_scheduler["eta_min"],
                    verbose=self.verbose,
                )

            elif name_scheduler == "ExponentialLR":
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer,
                    gamma=dict_params_scheduler["gamma"],
                    verbose=self.verbose,
                )

            else:
                raise ValueError(
                    'Type of scheduler {} unknown, only "MultiStepLR", "ExponentialLR", "CosineAnnealingLR" or "CosineAnnealingWarmRestarts".'.format(
                        name_scheduler
                    )
                )

        # if warmup_duration is not None:
        #    self.scheduler = create_lr_scheduler_with_warmup(scheduler,
        #                                                     warmup_start_value=1e-7,
        #                                                     warmup_end_value=learning_rate,
        #                                                     warmup_duration=warmup_duration)
        # else:
        #    self.scheduler = scheduler

        if self.all_gpu:
            # ===========dummy forward to intialize Lazy Module=========== #
            self.model.to("cpu")
            for ts in train_loader:
                self.model(torch.rand(ts.shape))
                break
            # ===========data Parrallel Module call=========== #
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        if self.path_checkpoint is not None:
            self.path_checkpoint = path_checkpoint
        else:
            self.path_checkpoint = os.getcwd() + os.sep + "model"

        self.log = {}
        self.train_time = 0
        self.passed_epochs = 0
        self.loss_train_history = []
        self.loss_valid_history = []

    def train(self, n_epochs=10):
        """
        Public function : master training loop over epochs
        """
        t = time.time()
        for epoch in range(n_epochs):
            # =======================one epoch===================== #
            train_loss = self.__train(epoch)
            self.loss_train_history.append(train_loss)

            if self.valid_loader is not None:
                valid_loss = self.__evaluate()
                self.loss_valid_history.append(valid_loss)

            # =======================verbose======================= #
            if self.verbose:
                logging.info("Epoch [{}/{}]".format(epoch + 1, n_epochs))
                logging.info("    Train loss : {:.6f}".format(train_loss))
                if self.valid_loader is not None:
                    logging.info("    Valid  loss : {:.6f}".format(valid_loss))

            if epoch % 5 == 0 or epoch == n_epochs - 1:
                # =========================log========================= #
                if self.save_only_core:
                    self.log = {
                        "model_state_dict": self.model.module.core.state_dict()
                        if self.device == "cuda" and self.all_gpu
                        else self.model.core.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss_train_history": self.loss_train_history,
                        "loss_valid_history": self.loss_valid_history,
                        "time": (time.time() - t),
                    }
                else:
                    self.log = {
                        "model_state_dict": self.model.module.state_dict()
                        if self.device == "cuda" and self.all_gpu
                        else self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss_train_history": self.loss_train_history,
                        "loss_valid_history": self.loss_valid_history,
                        "time": (time.time() - t),
                    }

                if self.save_checkpoint:
                    self.save()

            if self.scheduler is not None:
                if self.name_scheduler != "CosineAnnealingWarmRestarts":
                    self.scheduler.step()

            self.passed_epochs += 1

        self.train_time = round((time.time() - t), 3)

        if self.save_checkpoint:
            self.save()

        if self.plotloss:
            self.plot_history()

        return

    def save(self):
        """
        Public function : save log
        """
        torch.save(self.log, self.path_checkpoint + ".pt")
        return

    def plot_history(self):
        """
        Public function : plot loss history
        """
        plt.figure()
        plt.plot(range(self.passed_epochs), self.loss_train_history, label="Train loss")
        if self.valid_loader is not None:
            plt.plot(
                range(self.passed_epochs), self.loss_valid_history, label="Valid loss"
            )
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.legend()
        if self.save_fig:
            plt.savefig(self.path_fig)
        else:
            plt.show()
        return

    def reduce_lr(self, new_lr):
        for g in self.optimizer.param_groups:
            g["lr"] = new_lr
        return

    def __train(self, epoch):
        """
        Private function : model training loop over data loader
        """
        loss_train = 0
        iters = len(self.train_loader)

        for i, ts in enumerate(self.train_loader):
            self.model.train()
            # ===================variables=================== #
            ts = torch.Tensor(ts.float()).to(self.device)
            if self.mask is not None:
                mask_loss, ts_masked = self.mask(ts)
            # ===================forward===================== #
            self.optimizer.zero_grad()
            if self.mask is not None:
                outputs = self.model(ts_masked.to(self.device))
                loss = self.criterion(
                    outputs, ts.to(self.device), mask_loss.to(self.device)
                )
            else:
                if self.loss_in_model:
                    outputs, loss = self.model(ts.to(self.device))
                    loss = loss.mean()
                else:
                    outputs = self.model(ts.to(self.device))
                    loss = self.criterion(outputs, ts.to(self.device))
            # ===================backward==================== #
            loss.backward()
            self.optimizer.step()
            loss_train += loss.item()

            if self.name_scheduler == "CosineAnnealingWarmRestarts":
                self.scheduler.step(epoch + i / iters)

        loss_train = loss_train / len(self.train_loader)
        return loss_train

    def __evaluate(self):
        """
        Private function : model evaluation loop over data loader
        """
        loss_valid = 0
        with torch.no_grad():
            for ts in self.valid_loader:
                self.model.eval()
                # ===================variables=================== #
                ts = torch.Tensor(ts.float()).to(self.device)
                if self.mask is not None:
                    mask_loss, ts_masked = self.mask(ts)
                # ===================forward===================== #
                if self.mask is not None:
                    outputs = self.model(ts_masked.to(self.device))
                    loss = self.criterion(
                        outputs, ts.to(self.device), mask_loss.to(self.device)
                    )
                else:
                    if self.loss_in_model:
                        outputs, loss = self.model(ts.to(self.device))
                        loss = loss.mean()
                    else:
                        outputs = self.model(ts.to(self.device))
                        loss = self.criterion(outputs, ts.to(self.device))
                loss_valid += loss.item()

        loss_valid = loss_valid / len(self.valid_loader)
        return loss_valid


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
