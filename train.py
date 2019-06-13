import argparse
import yaml
from torch.utils.data import DataLoader
from torch import optim
from networks.tacotron import Tacotron

from dataset import TextSpeechDataset


def train():
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


def run_epoch():
    pass


def run_validate():
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
    train()
