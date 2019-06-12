from torch.nn import Module, ModuleList, GRUCell, Linear
from torch import stack, cat
from torch.autograd import Variable
from modules.prenet import Prenet
from modules.attention import AttentionRNN


class Decoder(Module):
    def __init__(self, in_features, memory_dim, r, eps=0, mode='train'):
        super(Decoder, self).__init__()
        self.mode = mode
        self.max_decoder_steps = 200
        self.memory_dim = memory_dim
        self.eps = eps
        self.r = r

        self.prenet = Prenet(in_features=(memory_dim * r), out_features=[256, 128])
        self.attention_rnn = AttentionRNN(out_dim=256, annot_dim=in_features, memory_dim=128)
        self.project_to_decoder_in = Linear(in_features=(256+in_features), out_features=256)
        self.decoder_rnns = ModuleList([GRUCell(256, 256), GRUCell(256, 256)])  # 2-layer residual GRU (256 cells)
        self.proj_to_mel = Linear(256, memory_dim * r)

    def forward(self, inputs, memory=None):
        B = inputs.size(0)
        greedy = not self.training
        if memory is not None:
            if memory.size(-1) == self.memory_dim:
                memory = memory.view(B, memory.size(1) // self.r, -1)
            T_decoder = memory.size(1)

        initial_memory = Variable(
            inputs.data.new(B, self.memory_dim * self.r).zero_())

        attention_rnn_hidden = Variable(inputs.data.new(B, 256).zero_())
        decoder_rnn_hiddens = [Variable(inputs.data.new(B, 256).zero_())
            for _ in range(len(self.decoder_rnns))]
        current_context_vec = Variable(inputs.data.new(B, 256).zero_())

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
                    memory_input = memory[t-1]

            processed_memory = self.prenet(memory_input)
            attention_rnn_hidden, current_context_vec, alignment = self.attention_rnn(
                processed_memory, current_context_vec, attention_rnn_hidden, inputs)
            decoder_input = self.project_to_decoder_in(
                cat((attention_rnn_hidden, current_context_vec), -1))
            for idx in range(len(self.decoder_rnns)):
                decoder_rnn_hiddens[idx] = self.decoder_rnns[idx](
                    decoder_input, decoder_rnn_hiddens[idx])
                # Residual connectinon
                decoder_input = decoder_rnn_hiddens[idx] + decoder_input
            output = decoder_input

            output = self.proj_to_mel(output)
            outputs += [output]
            alignments += [alignment]
            t += 1
            if (not greedy and self.training) or (greedy and memory is not None):
                if t >= T_decoder:
                    break
            else:
                if t > 1 and ((output.view(self.r, -1).data <= self.eps).prod(0) > 0).any() and alignment.data[:, int(alignment.shape[1]/2):].sum() > 0.7:
                    break
                elif t > self.max_decoder_steps:
                    break
        assert greedy or len(outputs) == T_decoder

        alignments = stack(alignments).transpose(0, 1)
        outputs = stack(outputs).transpose(0, 1).contiguous()
        return outputs, alignments
