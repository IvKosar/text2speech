import logging
import torch
from collections import defaultdict
import numpy as np
from tensorboardX import SummaryWriter

WINDOW_SIZE = 100


class MetricCounter:
    def __init__(self, exp_name):
        self.writer = SummaryWriter(exp_name)
        self.metrics = defaultdict(list)
        self.best_metric = float('inf')

    def clear(self):
        self.metrics = defaultdict(list)

    def add_losses(self, linear, mel, total):
        for name, value in zip(("linear_loss", "mel_loss", "total_loss"),
                               (linear, mel, total)):
            self.metrics[name].append(value)

    def loss_message(self):
        metrics = ((k, np.mean(self.metrics[k][-WINDOW_SIZE:])) for k in
                   ("linear_loss", "mel_loss", "total_loss"))
        return '; '.join(map(lambda x: x[0] + '=' + '%.5f' % x[1], metrics))

    def write_to_tensorboard(self, epoch_num, validation=False):
        scalar_prefix = 'Validation' if validation else 'Train'
        for k in ("linear_loss", "mel_loss", "total_loss"):
            self.writer.add_scalar(scalar_prefix + '_' + k, np.mean(self.metrics[k]), epoch_num)

    def write_audio_to_tensorboard(self, exp_name, inputs, outputs, targets, epoch_num, validation=False):
        pass

    def update_best_model(self):
        cur_metric = np.mean(self.metrics['total_loss'])
        if cur_metric < self.best_metric:
            self.best_metric = cur_metric
            return True
        return False
