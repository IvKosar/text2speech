import numpy as np


def pad_data(x, length):
    return np.pad(array=x, pad_width=(0, length - x.shape[0]), mode='constant', constant_values=0)


def prepare_tensor(inputs, out_steps):
    def pad_tensor(x, length):
        return np.pad(array=x, pad_width=[[0, 0], [0, length - x.shape[1]]], mode='constant', constant_values=0)

    max_len = max(list(map(lambda x: x.shape[1], inputs))) + 1
    remainder = max_len % out_steps
    pad_len = max_len + (out_steps - remainder) if remainder > 0 else max_len
    return np.stack(list(map(lambda x: pad_tensor(x, pad_len), inputs)))


def prepare_stop_target(inputs, out_steps):
    def pad_stop_target(x, length):
        return np.pad(array=x, pad_width=(0, length - x.shape[0]), mode='constant', constant_value=1.0)

    max_len = max(list(map(lambda x: x.shape[0], inputs))) + 1
    remainder = max_len % out_steps
    pad_len = max_len + (out_steps - remainder) if remainder > 0 else max_len
    return np.stack(list(map(lambda x: pad_stop_target(x, pad_len), inputs)))
