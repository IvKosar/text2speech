import torch.nn as nn
from torch.nn import functional as F
from torch import cat, bmm



class Attention(nn.Module):
    def __init__(self, query_dim, annot_dim, hidden_dim):
        super(Attention, self).__init__()
        self.query_layer = nn.Linear(query_dim, hidden_dim, bias=True)
        self.annot_layer = nn.Linear(annot_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, annots, query):
        if query.dim() == 2:
            # insert time-axis for broadcasting
            query = query.unsqueeze(1)
        # (batch, 1, dim)
        processed_query = self.query_layer(query)
        processed_annots = self.annot_layer(annots)

        # (batch, max_time)
        return self.v(F.tanh(processed_query + processed_annots)).squeeze(-1)


class AttentionRNN(nn.Module):
    def __init__(self, out_dim, annot_dim, memory_dim, score_mask_value=-float("inf")):
        super(AttentionRNN, self).__init__()
        self.rnn_cell = nn.GRUCell(out_dim + memory_dim, out_dim)
        self.alignment_model = Attention(query_dim=out_dim, annot_dim=annot_dim, hidden_dim=out_dim)
        self.score_mask_value = score_mask_value

    def forward(self, memory, context, rnn_state, annotations, mask=None, annotations_lengths=None):

        if annotations_lengths is not None and mask is None:
            mask = annotations.data.new(annotations.size(0), annotations.size(1)).byte().zero_()
            for idx, l in enumerate(annotations_lengths):
                mask[idx][:l] = 1
            mask = ~mask

        # Concat input query and previous context context
        rnn_input = cat((memory, context), -1)
        # rnn_input = rnn_input.unsqueeze(1)

        # Feed it to RNN
        # s_i = f(y_{i-1}, c_{i}, s_{i-1})
        rnn_output = self.rnn_cell(rnn_input, rnn_state)

        # Alignment
        # (batch, max_time)
        # e_{ij} = a(s_{i-1}, h_j)
        alignment = self.alignment_model(annotations, rnn_output)

        # Normalize context weight
        alignment = F.softmax(alignment, dim=-1)

        # Attention context vector
        # (batch, 1, dim)
        # c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j
        context = bmm(alignment.unsqueeze(1), annotations)
        context = context.squeeze(1)
        return rnn_output, context, alignment
