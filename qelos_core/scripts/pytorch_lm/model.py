import torch.nn as nn
import qelos_core as q


class NewRNNModel(nn.Module):
    encodertype = q.LSTMEncoder

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers,
                 dropout=0.5, tie_weights=False):
        super(NewRNNModel, self).__init__()
        worddic = dict(zip([str(x) for x in range(ntoken)], range(ntoken)))
        dims = [ninp] + [nhid] * nlayers
        self.nhid = nhid
        self.nlayers = nlayers

        self.dims = dims
        self.D = worddic
        self.states = None
        # make layers
        self.emb = q.WordEmb(dims[0], worddic=self.D)
        self.out = q.WordLinout(dims[-1], worddic=self.D)
        self.rnn = self.encodertype(*dims, bidir=False, bias=True, dropout_in=dropout, dropout_pt=dropout)
        self.rnn.ret_all_states = True

    def forward(self, x, states):   # (seqlen, batsize) and (numlayers, batsize, dim)
        x = x.transpose(1, 0)
        emb, xmask = self.emb(x)
        # do actual forward
        out, all_states = self.rnn._forward(emb, mask=xmask, states_0=states, ret_states=True)
        ret_states = zip(*all_states)
        # output
        out = self.out(out)
        out = out.transpose(1, 0).contiguous()
        return out, ret_states          # (seqlen, batsize, ntoken) and (numlayers, batsize, dim)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        ret = [[],[]]
        for i in range(self.nlayers):
            ret[0].append(weight.new_zeros(bsz, 1, self.nhid))
            ret[1].append(weight.new_zeros(bsz, 1, self.nhid))
        return ret


class Old_RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(Old_RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


# class RNNModel(Old_RNNModel):
class RNNModel(NewRNNModel):
    pass