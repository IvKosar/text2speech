import torch
from torch.nn import functional as F


class L1LossMasked():
    def __call__(self, inputs, targets, lengths):
        inputs = inputs.view(-1, inputs.shape[-1])
        targets_flat = targets.view(-1, targets.shape[-1])

        losses_flat = F.l1_loss(inputs=inputs, target=targets_flat, size_average=False, reduce=False)
        losses = losses_flat.view(*targets.size())
        # mask: (batch, max_len, 1)
        mask = sequence_mask(sequence_length=lengths, max_len=targets.size(1)).unsqueeze(2)
        losses *= mask.float()
        return losses.sum() / lengths.float().sum() * targets.shape[2]


def sequence_mask(sequence_length, max_len):
    with torch.no_grad():
        batch_size = sequence_length.size(0)
        seq_range = torch.arange(start=0, end=max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        if sequence_length.is_cuda:
            seq_range_expand = seq_range_expand.cuda()
        seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
        return seq_range_expand < seq_length_expand
