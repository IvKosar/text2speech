import argparse
import logging
import os
import shutil

import torch
import yaml
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import TextSpeechDataset
from loss import L1LossMasked
from metric_counter import MetricCounter
from networks.tacotron import Tacotron
from utils.utilities import lr_decay, check_update, save_checkpoint, create_experiment_folder, plot_spectrogram, \
    plot_alignment

use_cuda = torch.cuda.is_available()


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
        run_epoch(train_loader, optimizer, criterion, epoch)
        run_validate(val_loader, optimizer, criterion, metric_counter)
        if metric_counter.update_best_model():
            torch.save(model.state_dict(), os.path.join(os.path.join(WEIGHTS_SAVE_PATH,
                                                                     f"best_{configs['experiment_name']}")))
        torch.save(model.state_dict(), os.path.join(os.path.join(WEIGHTS_SAVE_PATH,
                                                                 f"last_{configs['experiment_name']}")))
        print(metric_counter.loss_message())
        logging.debug(
            f"Experiment Name: {configs['experiment_name']}, Epoch: {epoch}, Loss: {metric_counter.loss_message()}")


def run_epoch(model, dataloader, optimizer, criterion, epoch):
    avg_linear_loss = 0
    avg_mel_loss = 0

    n_priority_freq = int(3000 / (configs["audio"]["sample_rate"] * 0.5) * configs["audio"]["num_freq"])
    for num_iter, data in enumerate(dataloader):

        current_step = num_iter + args.restore_step + epoch * len(dataloader) + 1

        # learning rate
        current_lr = lr_decay(configs["train"]["lr"], current_step, configs["train"]["warmup_steps"])
        for params_group in optimizer.param_groups:
            params_group['lr'] = current_lr

        optimizer.zero_grad()

        text_input_var = Variable(data[0])
        mel_spec_var = Variable(data[3])
        mel_lengths_var = Variable(data[4])
        linear_spec_var = Variable(data[2], volatile=True)

        if use_cuda:
            text_input_var = text_input_var.cuda()
            mel_spec_var = mel_spec_var.cuda()
            mel_lengths_var = mel_lengths_var.cuda()
            linear_spec_var = linear_spec_var.cuda()

        # forward pass
        mel_output, linear_output, alignments = model.forward(text_input_var, mel_spec_var)

        # loss computation
        mel_loss = criterion(mel_output, mel_spec_var, mel_lengths_var)
        linear_loss = 0.5 * criterion(linear_output, linear_spec_var, mel_lengths_var) \
                      + 0.5 * criterion(linear_output[:, :, :n_priority_freq],
                                        linear_spec_var[:, :, :n_priority_freq],
                                        mel_lengths_var)
        loss = mel_loss + linear_loss

        loss.backward()
        grad_norm, skip_flag = check_update(model=model, grad_clip=0.5, grad_top=100)
        if skip_flag:
            optimizer.zero_grad()
            print("iteration skip")
            continue
        optimizer.step()

        avg_linear_loss += linear_loss.data[0]
        avg_mel_loss += mel_loss.data[0]

        # # Plot Training Iter Stats
        # tb.add_scalar('TrainIterLoss/TotalLoss', loss.data[0], current_step)
        # tb.add_scalar('TrainIterLoss/LinearLoss', linear_loss.data[0],
        #               current_step)
        # tb.add_scalar('TrainIterLoss/MelLoss', mel_loss.data[0], current_step)
        # tb.add_scalar('Params/LearningRate', optimizer.param_groups[0]['lr'],
        #               current_step)
        # tb.add_scalar('Params/GradNorm', grad_norm, current_step)

        if current_step % configs["train"]["save_step"] == 0:
            if configs["train"]["checkpoint"]:
                # save model
                save_checkpoint(model, optimizer, linear_loss.data[0],
                                OUT_PATH, current_step, epoch)

            # Diagnostic visualizations
            const_spec = linear_output[0].data.cpu().numpy()
            gt_spec = linear_spec_var[0].data.cpu().numpy()

            const_spec = plot_spectrogram(const_spec, dataloader.dataset.ap)
            gt_spec = plot_spectrogram(gt_spec, dataloader.dataset.ap)
            # tb.add_image('Visual/Reconstruction', const_spec, current_step)
            # tb.add_image('Visual/GroundTruth', gt_spec, current_step)

            align_img = alignments[0].data.cpu().numpy()
            align_img = plot_alignment(align_img)
            # tb.add_image('Visual/Alignment', align_img, current_step)

            # Sample audio
            audio_signal = linear_output[0].data.cpu().numpy()
            dataloader.dataset.ap.griffin_lim_iters = 60
            audio_signal = dataloader.dataset.ap.inv_spectrogram(
                audio_signal.T)
            try:
                pass
                # tb.add_audio('SampleAudio', audio_signal, current_step,
                #              sample_rate=c.sample_rate)
            except:
                # print("\n > Error at audio signal on TB!!")
                # print(audio_signal.max())
                # print(audio_signal.min())
                pass

    avg_linear_loss /= (num_iter + 1)
    avg_mel_loss /= (num_iter + 1)
    avg_total_loss = avg_mel_loss + avg_linear_loss

    # # Plot Training Epoch Stats
    # tb.add_scalar('TrainEpochLoss/TotalLoss', avg_total_loss, current_step)
    # tb.add_scalar('TrainEpochLoss/LinearLoss', avg_linear_loss, current_step)
    # tb.add_scalar('TrainEpochLoss/MelLoss', avg_mel_loss, current_step)


def run_validate(dataloader, optimizer, criterion, metric_counter):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--resume", type=str, help="Path to weights to resume from")
    args = parser.parse_args()

    with open(args["config"]) as f:
        configs = yaml.load(f)

    OUT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), configs["train"]["output_path"])
    OUT_PATH = create_experiment_folder(OUT_PATH)
    CHECKPOINT_PATH = os.path.join(OUT_PATH, 'checkpoints')
    shutil.copyfile(args.config_path, os.path.join(OUT_PATH, 'config.json'))

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
