import torch.nn as nn
from torch import cat, bmm, tanh
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, query_dim, annot_dim, hidden_dim):
        super(Attention, self).__init__()
        self.query_layer = nn.Linear(in_features=query_dim, out_features=hidden_dim, bias=True)
        self.annot_layer = nn.Linear(in_features=annot_dim, out_features=hidden_dim, bias=True)
        self.v = nn.Linear(in_features=hidden_dim, out_features=1, bias=False)

    def forward(self, annots, query):
        if query.dim() == 2:
            # insert time-axis for broadcasting
            query = query.unsqueeze(dim=1)
        # (batch, 1, dim)
        processed_query = self.query_layer(input=query)
        processed_annots = self.annot_layer(input=annots)

        # (batch, max_time)
        return self.v(tanh(processed_query + processed_annots)).squeeze(-1)


class AttentionRNN(nn.Module):
    def __init__(self, out_dim, annot_dim, memory_dim, score_mask_value=-float("inf")):
        super(AttentionRNN, self).__init__()
        self.rnn_cell = nn.GRUCell(input_size=(out_dim + memory_dim), hidden_size=out_dim, bias=True)
        self.alignment_model = Attention(query_dim=out_dim, annot_dim=annot_dim, hidden_dim=out_dim)
        self.score_mask_value = score_mask_value

    def forward(self, memory, context, rnn_state, annotations, mask=None, annotations_lengths=None):
        rnn_output = self.rnn_cell(input=cat(tensors=(memory, context), dim=-1), hx=rnn_state)

        # Alignment and context weight normalization
        alignment = F.softmax(input=self.alignment_model(annots=annotations, query=rnn_output), dim=-1)

        # Attention context vector
        context = bmm(input=alignment.unsqueeze(1), mat2=annotations).squeeze(1)
        return rnn_output, context, alignment
