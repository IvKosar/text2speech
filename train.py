import argparse
import logging
import os

import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TextSpeechDataset
from loss import L1LossMasked
from metric_counter import MetricCounter
from networks.tacotron import Tacotron
from utils.utilities import lr_decay

use_cuda = torch.cuda.is_available()


def prepare_directories():
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(LOG_DIR)
        os.makedirs(WEIGHTS_SAVE_PATH)


def train():
    metric_counter = MetricCounter(configs["experiment_name"])

    parameters = dict(audio_configs)
    parameters["text_cleaner"] = configs["text_cleaner"]
    train_dataset = TextSpeechDataset(data_configs["data_path"], data_configs["annotations_train"], parameters)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn,
                              num_workers=num_workers, drop_last=False, pin_memory=True)

    val_dataset = TextSpeechDataset(data_configs["data_path"], data_configs["annotations_val"], parameters)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, num_workers=num_workers,
                            collate_fn=val_dataset.collate_fn, drop_last=False, pin_memory=True)

    model = Tacotron(configs.pop("embedding_size"),
                     audio_configs["frequency"],
                     audio_configs["mels_size"],
                     configs.pop("r"))

    optimizer = optim.Adam(model.parameters(), lr=train_configs["lr"])
    criterion = L1LossMasked()

    if args.resume:
        model.load_state_dict(torch.load(args.resume))

    n_priority_freq = int(3000 / (configs["audio"]["sample_rate"] * 0.5) * configs["audio"]["frequency"])
    for epoch in range(train_configs["epochs"]):
        run_epoch(model, train_loader, optimizer, criterion, metric_counter, epoch, n_priority_freq)
        run_validate(model, val_loader, criterion, metric_counter, n_priority_freq)
        if metric_counter.update_best_model():
            torch.save(model.state_dict(), os.path.join(os.path.join(WEIGHTS_SAVE_PATH,
                                                                     f"best_{configs['experiment_name']}")))
        torch.save(model.state_dict(), os.path.join(os.path.join(WEIGHTS_SAVE_PATH,
                                                                 f"last_{configs['experiment_name']}")))
        print(metric_counter.loss_message())
        logging.debug(
            f"Experiment Name: {configs['experiment_name']}, Epoch: {epoch}, Loss: {metric_counter.loss_message()}")


def run_epoch(model, dataloader, optimizer, criterion, metric_counter, epoch, n_priority_freq):
    for num_iter, data in enumerate(dataloader):
        current_step = num_iter + epoch * len(dataloader) + 1
        current_lr = lr_decay(configs["train"]["lr"], current_step, configs["train"]["warmup_steps"])
        for params_group in optimizer.param_groups:
            params_group['lr'] = current_lr

        optimizer.zero_grad()

        texts = data[0]
        linears = data[1]
        mels = data[2]
        mel_lengths = data[3]

        if use_cuda:
            texts = texts.cuda()
            mels = mels.cuda()
            mel_lengths = mel_lengths.cuda()
            linears = linears.cuda()

        # forward pass
        mel_output, linear_output, alignments = model.forward(texts, mels)

        # loss computation
        mel_loss = criterion(mel_output, mels, mel_lengths)
        linear_loss = 0.5 * criterion(linear_output, linears, mel_lengths) \
                      + 0.5 * criterion(linear_output[:, :, :n_priority_freq],
                                        linears[:, :, :n_priority_freq],
                                        mel_lengths)
        loss = mel_loss + linear_loss
        loss.backward()
        optimizer.step()


def run_validate(model, dataloader, criterion, metric_counter, n_priority_freq):
    model = model.eval()
    avg_linear_loss = 0
    avg_mel_loss = 0
    iter = 0
    for data in tqdm(dataloader):
        with torch.no_grad():
            texts, linears, mels, mel_lengths = data
            if use_cuda:
                texts = texts.cuda()
                linears = linears.cuda()
                mels = mels.cuda()
                mel_lengths = mel_lengths.cuda()

            mel_outputs, linear_outputs, alignments = model.forward(texts, mels)
            linear_loss = 0.5 * criterion(linear_outputs, linears, mel_lengths) \
                          + 0.5 * criterion(linear_outputs[:, :, :n_priority_freq],
                                            linears[:, :, :n_priority_freq], mel_lengths)
            mel_loss = criterion(mel_outputs, mels, mel_lengths)

            avg_linear_loss += linear_loss.item()
            avg_mel_loss += mel_loss.item()
            iter += 1

    avg_linear_loss /= (iter + 1)
    avg_mel_loss /= (iter + 1)
    avg_total_loss = avg_mel_loss + avg_linear_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--resume", type=str, help="Path to weights to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        configs = yaml.load(f)

    batch_size = configs.pop("batch_size")
    eval_batch_size = configs.pop("eval_batch_size")
    num_workers = configs.pop("num_workers")
    data_configs = configs.pop("data")
    audio_configs = configs.pop("audio")
    train_configs = configs.pop("train")

    output_path = data_configs["output_path"]
    LOG_DIR = os.path.join(output_path, configs["experiment_name"], "logs")
    WEIGHTS_SAVE_PATH = os.path.join(output_path, configs["experiment_name"], "weights")

    train()
