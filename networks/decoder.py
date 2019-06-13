from torch import stack, cat
from torch.autograd import Variable
from torch.nn import Module, ModuleList, GRUCell, Linear

from modules.attention import AttentionRNN
from modules.prenet import Prenet


class Decoder(Module):
    def __init__(self, in_features, memory_dim, r, eps=0):
        super(Decoder, self).__init__()
        self.max_decoder_steps = 200
        self.memory_dim = memory_dim
        self.eps = eps
        self.r = r

        self.prenet = Prenet(in_features=(self.memory_dim * self.r), out_features=[256, 128])
        self.attention_rnn = AttentionRNN(out_dim=256, annot_dim=in_features, memory_dim=128)
        self.project_to_decoder_in = Linear(in_features=(256 + in_features), out_features=256)
        # 2-layer residual GRU (256 cells)
        self.decoder_rnns = ModuleList(
            [GRUCell(input_size=256, hidden_size=256), GRUCell(input_size=256, hidden_size=256)])
        self.proj_to_mel = Linear(in_features=256, out_features=(self.memory_dim * self.r))

    def forward(self, inputs, memory=None):
        B = inputs.size(0)
        greedy = not self.training
        if memory is not None:
            if memory.size(-1) == self.memory_dim:
                memory = memory.view(B, memory.size(1) // self.r, -1)
            T_decoder = memory.size(1)

        initial_memory = Variable(tensor=inputs.data.new(B, self.memory_dim * self.r).zero_())

        attention_rnn_hidden = Variable(tensor=inputs.data.new(B, 256).zero_())
        decoder_rnn_hiddens = [Variable(tensor=inputs.data.new(B, 256).zero_()) for _ in range(len(self.decoder_rnns))]
        current_context_vec = Variable(tensor=inputs.data.new(B, 256).zero_())

        if memory is not None:
            memory = memory.transpose(0, 1)
        outputs = []
        alignments = []
        t = 0
        memory_input = initial_memory
        while True:
            if t > 0:
                if greedy:
                    memory_input = outputs[-1]
                else:
                    memory_input = memory[t - 1]

            processed_memory = self.prenet(memory_input)
            attention_rnn_hidden, current_context_vec, alignment = self.attention_rnn(memory=processed_memory,
                                                                                      context=current_context_vec,
                                                                                      rnn_state=attention_rnn_hidden,
                                                                                      annotations=inputs)
            decoder_input = self.project_to_decoder_in(
                input=cat(tensors=(attention_rnn_hidden, current_context_vec), dim=-1))
            for idx in range(len(self.decoder_rnns)):
                decoder_rnn_hiddens[idx] = self.decoder_rnns[idx](input=decoder_input, hx=decoder_rnn_hiddens[idx])
                # Residual connectinon
                decoder_input = decoder_rnn_hiddens[idx] + decoder_input
            output = decoder_input

            output = self.proj_to_mel(input=output)
            outputs += [output]
            alignments += [alignment]
            t += 1
            if (not greedy and self.training) or (greedy and memory is not None):
                if t >= T_decoder:
                    break
            else:
                if t > 1 and ((output.view(self.r, -1).data <= self.eps).prod(0) > 0).any() and alignment.data[:, int(
                        alignment.shape[1] / 2):].sum() > 0.7:
                    break
                elif t > self.max_decoder_steps:
                    break
        assert greedy or len(outputs) == T_decoder

        alignments = stack(alignments).transpose(0, 1)
        outputs = stack(outputs).transpose(0, 1).contiguous()
        return outputs, alignments
