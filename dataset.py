from torch.utils.data import Dataset


class TextSpeechDataset(Dataset):
    def __init__(self, root_dir, parameters):
        """
        :param root_dir: path to data
        :param parameters: dict with parameters
        """
        self.root_dir = root_dir
        self.cleaners = parameters["text_cleaner"]
        self.outputs_per_step = parameters["outputs_per_step"]
        self.sample_rate = parameters["sample_rate"]
        self.min_seq_len = parameters["min_seq_len"]

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass
