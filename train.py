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
    prepare_directories()
    metric_counter = MetricCounter(exp_name=LOG_DIR)

    parameters = dict(audio_configs)
    parameters["text_cleaner"] = configs["text_cleaner"]
    parameters["outputs_per_step"] = configs["r"]
    train_dataset = TextSpeechDataset(root_dir=data_configs["data_path"],
                                      annotations_file=data_configs["annotations_train"], parameters=parameters)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=train_dataset.collate_fn,
                              num_workers=num_workers, drop_last=False, pin_memory=True)

    val_dataset = TextSpeechDataset(root_dir=data_configs["data_path"],
                                    annotations_file=data_configs["annotations_val"], parameters=parameters)
    val_loader = DataLoader(dataset=val_dataset, batch_size=eval_batch_size, num_workers=num_workers,
                            collate_fn=val_dataset.collate_fn, drop_last=False, pin_memory=True)

    model = Tacotron(embedding_dim=configs.pop("embedding_size"),
                     linear_dim=audio_configs["frequency"],
                     mel_dim=audio_configs["mels_size"],
                     r=configs.pop("r"))

    if use_cuda:
        model = torch.nn.DataParallel(model.to("cuda"))

    optimizer = optim.Adam(params=model.parameters(), lr=train_configs["lr"])
    criterion = L1LossMasked()

    if args.resume:
        model.load_state_dict(torch.load(args.resume))

    n_priority_freq = int(3000 / (audio_configs["sample_rate"] * 0.5) * audio_configs["frequency"])
    for epoch in range(train_configs["epochs"]):
        audio_signal = run_epoch(model, train_loader, optimizer, criterion, metric_counter, epoch, n_priority_freq)
        run_validate(model, val_loader, criterion, metric_counter, n_priority_freq)
        if metric_counter.update_best_model():
            torch.save(model.state_dict(), os.path.join(os.path.join(WEIGHTS_SAVE_PATH,
                                                                     f"best_{configs['experiment_name']}.pth.tar")))
            audio_signal = train_dataset.ap.spectrogram_to_wav(audio_signal.T)
            metric_counter.write_audio_to_tensorboard("Audio", audio_signal, epoch, audio_configs["sample_rate"])

        torch.save(model.state_dict(), os.path.join(os.path.join(WEIGHTS_SAVE_PATH,
                                                                 f"last_{configs['experiment_name']}.pth.tar")))
        print(metric_counter.loss_message())
        logging.debug(
            f"Experiment Name: {configs['experiment_name']}, Epoch: {epoch}, Loss: {metric_counter.loss_message()}")


def run_epoch(model, dataloader, optimizer, criterion, metric_counter, epoch, n_priority_freq):
    model = model.train()
    num_iter = 0
    for data in tqdm(dataloader):
        current_step = num_iter + epoch * len(dataloader) + 1
        current_lr = lr_decay(init_lr=train_configs["lr"], global_step=current_step,
                              warmup_steps=train_configs["warmup_steps"])
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
        mel_loss = criterion(inputs=mel_output, targets=mels, lengths=mel_lengths)
        linear_loss = 0.5 * criterion(inputs=linear_output, targets=linears, lengths=mel_lengths) \
                      + 0.5 * criterion(linear_output[:, :, :n_priority_freq],
                                        linears[:, :, :n_priority_freq],
                                        mel_lengths)
        total_loss = mel_loss + linear_loss
        total_loss.backward()
        optimizer.step()

        metric_counter.add_losses(linear_loss.item(), mel_loss, total_loss)
        metric_counter.write_to_tensorboard(current_step)

        num_iter += 1
        return linear_output[0].data.cpu().numpy()


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

            mel_outputs, linear_outputs, alignments = model.forward(characters=texts, mel_specs=mels)
            linear_loss = 0.5 * criterion(inputs=linear_outputs, targets=linears, lengths=mel_lengths) \
                          + 0.5 * criterion(inputs=linear_outputs[:, :, :n_priority_freq],
                                            targets=linears[:, :, :n_priority_freq], lengths=mel_lengths)
            mel_loss = criterion(inputs=mel_outputs, targets=mels, lengths=mel_lengths)

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
        configs = yaml.load(f, Loader=yaml.FullLoader)

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
