import logging
from tqdm import tqdm
from time import time
from typing import Callable
import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau as BaseReduceLROnPlateau
from utils import logger, metrics
import optim
from optim import *


class Alchemist:
    def __init__(self, model, gpu=0):
        self.model = model
        self.device = self.get_device(gpu)
        self.model.to(self.device)

    def train(self, train_gen, val_gen, cfg, logdir):
        train_logger = logger.TrainLogger(logdir, benchmark=cfg.monitor, mode=cfg.monitor_mode)
        logging.info(f"Save checkpoints to {os.path.abspath(train_logger.checkpoint)}")
        loss_fn = self.get_loss_fn(cfg.loss)
        optimizer = self.get_optimizer(cfg.optim)
        scheduler = self.get_scheduler(cfg, optimizer)

        logging.info(f"Start training: {len(train_gen)} batches/epoch")

        for epoch in range(cfg.epoch):
            start_time = time()
            epoch_loss = self.train_one_epoch(epoch, train_gen, loss_fn, optimizer, cfg, logger=train_logger,
                                              val_gen=val_gen)
            logging.info(f"[Epoch {epoch} loss]: {epoch_loss}, time: {(time() - start_time) / 60:.2f} min")
            train_logger.log(epoch, {'loss': epoch_loss}, domain='train')

            val_metrics = self.evaluate(val_gen, cfg.metrics)
            train_logger.log(epoch, val_metrics, domain='val')

            logging.info(
                f'[Epoch {epoch} Val Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_metrics.items()))

            monitor_value = train_logger.get_monitor_value(val_metrics)
            scheduler.step(monitor_value)
            early_stop = train_logger.checkpoint_and_earlystop(epoch, val_metrics, self.model.state_dict())
            if early_stop:
                logging.info(f"Early stop at epoch {epoch}")
                break
        logging.info(
            f"Load best checkpoint at epoch {train_logger.best_epoch} from {os.path.abspath(train_logger.checkpoint)}")
        self.model.load_weights(train_logger.checkpoint)

        # delete the checkpoint file
        if not cfg.save_checkpoint:
            train_logger.delete_checkpoint()

        logging.info(f"Training finished")

    def train_one_epoch(self, epoch, train_gen, loss_fn, optimizer, cfg, logger=None, val_gen=None):
        self.model.train()
        total_loss = 0
        with tqdm(train_gen, ncols=120) as pbar:
            for batch_index, batch_data in enumerate(pbar):
                if cfg.optim.optimizer.lower() == "helen":
                    batch_loss = self.train_one_batch_Helen(batch_data, loss_fn, optimizer)
                else:
                    batch_loss = self.train_one_batch(batch_data, loss_fn, optimizer, cfg.optim.max_grad_norm)

                total_batch = batch_index + epoch * len(train_gen)
                if cfg.verbose > 0 and total_batch % cfg.verbose == 0:
                    # self.record_weight_gradient_norm(logger, total_batch)
                    logger.log(total_batch, {'total_loss': batch_loss, 'logloss': self._logloss, 'reg_loss': self._reg,
                                             'embed_reg': self.model._embed_reg, 'net_reg': self.model._net_reg},
                               domain='train_iteration')

                    val_metrics = self.evaluate(val_gen, cfg.metrics)
                    logger.log(total_batch, val_metrics, domain='val_iteration')

                total_loss += batch_loss

        return total_loss / len(train_gen)

    def train_one_batch(self, batch_data, loss_fn, optimizer, max_grad_norm=None):
        X, y = self.inputs_to_device(batch_data)
        loss = self.calculate_total_loss(self.model(X), y, loss_fn)
        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        optimizer.step()
        return loss.item()

    def train_one_batch_Helen(self, batch_data, loss_fn, optimizer):
        X, y = self.inputs_to_device(batch_data)

        optimizer.count_feature_occurrence(X, self.model.get_feature_params_map(), self.model.feature_specs)

        loss = self.calculate_total_loss(self.model(X), y, loss_fn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.first_step(zero_grad=True)
        loss = self.calculate_total_loss(self.model(X), y, loss_fn)
        loss.backward()
        optimizer.second_step()
        return loss.item()

    def evaluate(self, val_gen, all_metrics):
        self.model.eval()
        with torch.no_grad():
            y_pred = []
            y_true = []
            with tqdm(val_gen, ncols=120) as pbar:
                for batch_index, batch_data in enumerate(pbar):
                    X, y = self.inputs_to_device(batch_data)
                    pred = self.model(X)
                    y_pred.extend(pred.cpu().numpy().reshape(-1))
                    y_true.extend(y.cpu().numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            y_true = np.array(y_true, np.float64)
            val_metrics = metrics.evaluate_metrics(y_true, y_pred, all_metrics)
        return val_metrics

    def calculate_total_loss(self, pred, label, loss_fn):
        loss = loss_fn(pred, label)
        reg = self.model.regularizer()
        # cache these losses for logging
        self._logloss = loss.item()
        self._reg = reg.item()
        return loss + reg

    def inputs_to_device(self, inputs):
        X, y = inputs
        X = X.to(self.device)
        y = y.to(self.device)
        return X, y

    @staticmethod
    def get_device(gpu=-1):
        if gpu >= 0 and torch.cuda.is_available():
            device = torch.device("cuda:" + str(gpu))
        else:
            device = torch.device("cpu")
        return device

    def get_optimizer(self, cfg_optim):
        opt = cfg_optim.optimizer
        embed_params = self.model.embed_params()
        net_params = self.model.net_params()
        lr_net = cfg_optim.lr_net
        lr_embed = cfg_optim.lr_embed
        params = embed_params + net_params

        param_group = [
            {'params': embed_params, 'lr': lr_embed, 'embed': True},
            {'params': net_params, 'lr': lr_net, 'embed': False}
        ]

        if isinstance(opt, str):
            if opt.lower() == "adam":
                opt = "Adam"
        if opt.lower() == "adam":
            logging.info("Using Adam optimizer")
            opt = optim.Adam(param_group, betas=cfg_optim.betas)
        elif opt.lower() == "helen":
            logging.info("Using contest or Helen optimizer")
            opt_class = eval(opt)
            opt = opt_class(embed_params, net_params, **cfg_optim)
        else:
            try:
                logging.info("Using {} optimizer".format(opt))
                opt = getattr(torch.optim, opt)(param_group)
            except:
                raise NotImplementedError("optimizer={} is not supported.".format(opt))
        return opt

    @staticmethod
    def get_scheduler(cfg, optimizer):
        if cfg.optim.lr_decay and not cfg.optim.warmup_steps > 0:
            scheduler = ReduceLROnPlateau(optimizer, mode=cfg.monitor_mode,
                                          factor=cfg.optim.gamma, verbose=True,
                                          patience=0, min_lr=cfg.optim.min_lr,
                                          threshold=1e-6, threshold_mode='abs')

        elif cfg.optim.lr_decay and cfg.optim.warmup_steps > 0:
            scheduler = WarmupReduceLROnPlateau(optimizer, warmup_steps=cfg.optim.warmup_steps,
                                                mode=cfg.monitor_mode,
                                                factor=cfg.optim.gamma, verbose=True,
                                                patience=0, min_lr=cfg.optim.min_lr,
                                                threshold=1e-6, threshold_mode='abs')
        elif not cfg.optim.lr_decay and cfg.optim.warmup_steps > 0:
            scheduler = WarmupLR(optimizer, warmup_steps=cfg.optim.warmup_steps)

        else:
            class FakeScheduler(object):
                def step(self, *args, **kwargs):
                    pass

            scheduler = FakeScheduler()

        return scheduler

    @staticmethod
    def get_loss_fn(loss):
        if isinstance(loss, str):
            if loss in ["bce", "binary_crossentropy", "binary_cross_entropy"]:
                loss = "binary_cross_entropy"
        try:
            loss_fn = getattr(torch.functional.F, loss)
        except:
            try:
                from . import losses
                loss_fn = getattr(losses, loss)
            except:
                raise NotImplementedError("loss={} is not supported.".format(loss))
        return loss_fn


class ReduceLROnPlateau(BaseReduceLROnPlateau):
    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else
                                 "%.5d") % epoch
                    logging.info('Epoch {}: reducing learning rate'
                                 ' of group {} to {:.4e}.'.format(epoch_str, i, new_lr))


class WarmupLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupLR, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, epoch):
        if epoch < self.warmup_steps:
            return float(epoch) / float(max(1, self.warmup_steps))
        else:
            return 1.0


class WarmupReduceLROnPlateau(ReduceLROnPlateau):
    """
    Optimizer scheduler that combines learning rate warmup with ReduceLROnPlateau.
    if warmup_epochs > 0, the learning rate will be increased linearly from 0 to the initial learning rate.
    else, no warmup will be performed.
    """

    def __init__(self, optimizer, warmup_steps: int, *args, **kwargs):
        self.warmup_epochs = warmup_steps
        self.current_step = 0
        super(WarmupReduceLROnPlateau, self).__init__(optimizer, *args, **kwargs)

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            lr = min(1.0, self.current_step / self.warmup_steps)
            return [base_lr * lr for base_lr in self.base_lrs]
        else:
            return super(WarmupReduceLROnPlateau, self).get_lr()

    def step(self, metrics=None):
        self.current_step += 1
        super(WarmupReduceLROnPlateau, self).step(metrics=metrics)
