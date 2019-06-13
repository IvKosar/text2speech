import os
import logging
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torch import optim

from dataset import TextSpeechDataset
from networks.tacotron import Tacotron
from loss import L1LossMasked
from metric_counter import MetricCounter


def prepare_directories():
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(LOG_DIR)
        os.makedirs(WEIGHTS_SAVE_PATH)


def train():
    metric_counter = MetricCounter(configs["experiment_name"])

    train_dataset = TextSpeechDataset(data_configs["data_path"], data_configs["annotations_train"], audio_configs)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn,
                              num_workers=num_workers, drop_last=False, pin_memory=True)

    val_dataset = TextSpeechDataset(data_configs["data_path"], data_configs["annotations_val"], audio_configs)
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

    for epoch in range(train_configs["epochs"]):
        run_epoch(train_loader, optimizer, criterion, metric_counter)
        run_validate(val_loader, optimizer, criterion, metric_counter)
        if metric_counter.update_best_model():
            torch.save(model.state_dict(), os.path.join(os.path.join(WEIGHTS_SAVE_PATH,
                                                                     f"best_{configs['experiment_name']}")))
        torch.save(model.state_dict(), os.path.join(os.path.join(WEIGHTS_SAVE_PATH,
                                                                 f"last_{configs['experiment_name']}")))
        print(metric_counter.loss_message())
        logging.debug(
            f"Experiment Name: {configs['experiment_name']}, Epoch: {epoch}, Loss: {metric_counter.loss_message()}")


def run_epoch(dataloader, optimizer, criterion, metric_counter):
    pass


def run_validate(dataloader, optimizer, criterion, metric_counter):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--resume", type=str, help="Path to weights to resume from")
    args = parser.parse_args()

    with open(args["config"]) as f:
        configs = yaml.load(f)

    batch_size = configs.pop("batch_size")
    eval_batch_size = configs.pop("eval_batch_size")
    num_workers = configs.pop("num_workers")
    data_configs = configs.pop("data")
    audio_configs = configs.pop("audio")
    train_configs = configs.pop("train")

    output_path = data_configs["output_path"]
    LOG_DIR = os.path.join(output_path, configs.pop("experiment_name"), "logs")
    WEIGHTS_SAVE_PATH = os.path.join(output_path, configs.pop("experiment_name"), "weights")

    train()
