import numpy as np
from torch.nn.utils import clip_grad_norm
import os
import datetime
from collections import OrderedDict
from torch import save


import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

def check_update(model, grad_clip, grad_top):
    """
    Check gradient against limits
    """
    skip_flag = False
    grad_norm = clip_grad_norm(parameters=model.parameters(), max_norm=grad_clip)
    if np.isinf(x=grad_norm):
        print("gradient is inf")
        skip_flag = True
    elif grad_norm > grad_top:
        print("gradient is above the top level")
        skip_flag = True
    return grad_norm, skip_flag


def lr_decay(init_lr, global_step, warmup_steps):
    warmup_steps = float(warmup_steps)
    step = global_step + 1.
    lr = init_lr * warmup_steps ** 0.5 * np.minimum(step * warmup_steps ** -1.5,
                                                    step ** -0.5)
    return lr


def save_checkpoint(model, optimizer, model_loss, out_path,
                    current_step, epoch):
    checkpoint_path = 'checkpoint_{}.pth.tar'.format(current_step)
    checkpoint_path = os.path.join(out_path, checkpoint_path)
    print("\n | > Checkpoint saving : {}".format(checkpoint_path))

    new_state_dict = _trim_model_state_dict(model.state_dict())
    state = {'model': new_state_dict,
             'optimizer': optimizer.state_dict(),
             'step': current_step,
             'epoch': epoch,
             'linear_loss': model_loss,
             'date': datetime.date.today().strftime("%B %d, %Y")}
    save(state, checkpoint_path)



def _trim_model_state_dict(state_dict):
    r"""Remove 'module.' prefix from state dictionary. It is necessary as it
    is loded for the next time by model.load_state(). Otherwise, it complains
    about the torch.DataParallel()"""

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict



def create_experiment_folder(root_path):
    """ Create a folder with the current date and time """
    date_str = datetime.datetime.now().strftime("%B-%d-%Y_%I:%M%p")
    output_folder = os.path.join(root_path, date_str)
    os.makedirs(output_folder, exist_ok=True)
    print(" > Experiment folder: {}".format(output_folder))
    return output_folder


def plot_alignment(alignment, info=None):
    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(alignment.T, aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_spectrogram(linear_output, audio):
    spectrogram = audio._denormalize(linear_output)
    fig = plt.figure(figsize=(16, 10))
    plt.imshow(spectrogram.T, aspect="auto", origin="lower")
    plt.colorbar()
    plt.tight_layout()
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data
