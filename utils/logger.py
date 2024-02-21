import copy
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch

import logging

info = logging.info


class TrainLogger:
    def __init__(self, log_dir, checkpoint=None, benchmark='auc', mode='max', patience=2):
        """
        :param threshold: metrics / best_metrics > 1 + threshold, set the early stop flag true
        :param maxdown: Maximum number of epochs allowed where the metrics is going down
        :param benchmark: `p` | `map` | `ndcg` | `mrr` | `hit` | `r` | `f`
        """

        self.log_dir = log_dir
        self.writer = SummaryWriter(os.path.join(log_dir, 'tensorboard'))
        logging.info(f"Tensorboard log dir: {os.path.abspath(os.path.join(log_dir, 'tensorboard'))}")
        self.checkpoint = os.path.join(log_dir, 'checkpoint.pt') if checkpoint is None else checkpoint
        if isinstance(benchmark, str):
            self.benchmark = {benchmark.lower(): 1}
        else:
            self.benchmark = {k.lower(): v for k, v in benchmark.items()}

        self.mode = mode.lower()
        assert self.mode in ['min', 'max'], "mode must be 'min' or 'max'"
        self.patience = patience

        self.best_metrics = defaultdict(lambda: -float('inf') if self.mode == 'max' else float('inf'))
        self.best_epoch = -1
        self._stopping_steps = 0

        # for each iteration and each feature field, record the embedding gradient and corresponding occurence
        self.embedding_grad_norm_cache = []

    def log(self, epoch, metrics, domain='train'):
        for k, v in metrics.items():
            self.writer.add_scalar(f'{domain}/{k}', v, epoch)

    @torch.no_grad()
    def log_embedding_grad_norm_occurrence(self, idx, embed_norm_count_list):
        # self.embedding_grad_norm.append((idx, embed_norms, embedding_grad_norms, counts))
        for embed_idx, embed_norm_count in enumerate(embed_norm_count_list):
            self.writer.add_histogram(f'embedding_{embed_idx}/embedding_norm', embed_norm_count[0], idx)
            self.writer.add_histogram(f'embedding_{embed_idx}/embedding_grad_norm', embed_norm_count[1], idx)
            self.writer.add_histogram(f'embedding_{embed_idx}/embedding_occurrence', embed_norm_count[2], idx)

    @torch.no_grad()
    def log_embedding_grad_norm_occurrence_fig(self, idx, embed_norm_count_list):
        # plot the embedding gradient and corresponding occurence and add it to tensorboard
        for embed_idx, embed_norm_count in enumerate(embed_norm_count_list):
            # plot the embedding gradient and corresponding occurence
            occurence = embed_norm_count[2].cpu().numpy()
            embedding_grad_norm = embed_norm_count[1].cpu().numpy()
            embedding_norm = embed_norm_count[0].cpu().numpy()

            fig, ax = plt.subplots(2, 1, figsize=(10, 10))
            ax[0].hist2d(occurence, embedding_norm, bins=80, cmap='Blues')
            fig.colorbar(ax[0].collections[0], ax=ax[0])
            ax[0].set_xlabel('embedding occurrence')
            ax[0].set_ylabel('embedding norm')
            ax[0].set_title('embedding norm vs occurrence')
            ax[1].hist2d(occurence, embedding_grad_norm, bins=80, cmap='Blues')
            fig.colorbar(ax[1].collections[0], ax=ax[1])
            ax[1].set_xlabel('embedding occurrence')
            ax[1].set_ylabel('embedding grad norm')
            ax[1].set_title('embedding grad norm vs occurrence')
            self.writer.add_figure(f'embedding_{embed_idx}/embedding_grad_norm_occurrence', fig, idx)

    def checkpoint_and_earlystop(self, epoch, metrics, state_dict, min_delta=1e-6):
        monitor_value = self.get_monitor_value(metrics)
        best_value = self.get_monitor_value(self.best_metrics)

        is_best = False
        if self.mode == 'max' and monitor_value > best_value + min_delta:
            is_best = True
        elif self.mode == 'min' and monitor_value < best_value - min_delta:
            is_best = True
        elif epoch == 0:
            is_best = True

        if is_best:
            self._update_best_record(epoch, metrics)
            self.save_checkpoint(state_dict)
            logging.info(f"Save checkpoint to {os.path.abspath(self.checkpoint)}")
            self._stopping_steps = 0
        else:
            self._stopping_steps += 1

        return self._stopping_steps >= self.patience

    def get_monitor_value(self, metrics):
        value = 0
        for k, v in self.benchmark.items():
            value += metrics.get(k, 0) * v
        return value

    def save_checkpoint(self, state_dict):
        torch.save(state_dict, self.checkpoint)

    def delete_checkpoint(self):
        if os.path.exists(self.checkpoint):
            os.remove(self.checkpoint)

    def _update_best_record(self, epoch, metrics):
        for k, v in metrics.items():
            self.best_metrics[k] = v
        self.best_epoch = epoch

    def __del__(self):
        self.writer.close()

    @staticmethod
    def info(info_str):
        logging.info(info_str)
