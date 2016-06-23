from abc import abstractmethod

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import serializers

from variable import Variable as xp

class BaseModel(chainer.Chain):
    @abstractmethod
    def __call__(self):
        pass

    def save_model(self, filename):
        pass

    def load_model(self, filename):
        pass

class Encoder(BaseModel):
    def __init__(self, embed_size, hidden_size):
        super().__init__(
            # input weight vector of {input, output, forget} gate and input
            w = L.Linear(embed_size, 4 * hidden_size),
            # hidden weight vector of {input, output, forget} gate and input
            v = L.Linear(hidden_size, 4 * hidden_size),
        )

    def __call__(self, x, c, h):
        return F.lstm(c, self.w(x) + self.v(h))

class Decoder(BaseModel):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__(
            # embedding previous output
            ye = L.EmbedID(vocab_size, embed_size),
            # input(previous output) weight vector
            eh = L.Linear(embed_size, 4 * hidden_size),
            # hidden weight vector
            hh = L.Linear(hidden_size, 4 * hidden_size),
            # forward encoder weight vector
            ah = L.Linear(hidden_size, 4 * hidden_size),
            # backward encoder weight vector
            bh = L.Linear(hidden_size, 4 * hidden_size),
            # decoder weight weight vector
            hf = L.Linear(hidden_size, embed_size),
            # output weight vector
            fy = L.Linear(embed_size, vocab_size)
        )

    def __call__(self, y, c, h, a, b):
        e = F.tanh(self.ye(y))
        c, h = F.lstm(c, self.eh(e) + self.hh(h) + self.ah(a) + self.bh(b))
        f = F.tanh(self.hf(h))
        return self.fy(f), c, h

class Attention(BaseModel):
    def __init__(self, hidden_size):
        super().__init__(
            # forward encoder weight vector
            aw = L.Linear(hidden_size, hidden_size),
            # backward encoder weight vector
            bw = L.Linear(hidden_size, hidden_size),
            # previous hidden output weight vector
            pw = L.Linear(hidden_size, hidden_size),
            # attention output weight vector
            we = L.Linear(hidden_size, 1),
        )
        self.hidden_size = hidden_size

    def __call__(self, a_list, b_list, p):
        '''
        Args:
            p (chainer.Variable): previous output
        '''
        batch_size = p.data.shape[0]
        e_list = []
        sum_e = xp.zeros((batch_size, 1), dtype=xp.float32)
        for a, b in zip(a_list, b_list):
            w = F.tanh(self.aw(a) + self.bw(b) + self.pw(p))
            e = F.exp(self.we(w))
            e_list.append(e)
            sum_e += e
        aa = bb = xp.zeros((batch_size, self.hidden_size), dtype=xp.float32)
        for a, b, e in zip(a_list, b_list, e_list):
            e /= sum_e
            aa += F.reshape(F.batch_matmul(a, e), (batch_size, self.hidden_size))
            bb += F.reshape(F.batch_matmul(b, e), (batch_size, self.hidden_size))
        return aa, bb

class AttentionMT(BaseModel):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__(
            emb = EmbedID(vocab_size, embed_size),
            fenc = Encoder(embed_size, hidden_size),
            benc = Encoder(embed_size, hidden_size),
            att = Attention(hidden_size),
            dec = Decoder(vocab_size, embed_size, hidden_size)
        )
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

    def reset(self):
        self.zerograds()
        self.x_list = []

    def __call__(self, x, t, x_vocab):
        y = self.forward(x, t, x_vocab)
        loss = F.softmax_cross_entropy(y, t)
        return loss

    def forward(self, x, y, x_vocab):
        # preparing
        hidden_init = xp.zeros((self.hidden_size,))
        # embedding sentence
        x_list = [F.tanh(self.emb(w)) for w in x]
        # forward encoding
        a_list = []
        c, a = hidden_init
        for x in x_list:
            c, a = self.fenc(x, c, a)
            a_list.append(a)
        # backward encoding
        b_list = []
        c, b = hidden_init
        for x in x_list:
            c, b = self.benc(x, c, b)
            b_list.append(b)
        # attention
        h = hidden_init
        t = x_vocab[]
        p_list = []
        for _ in range(len(y)):
            aa, bb = self.att(a_list, b_list, h)
            p, c, h = self.dec(p, c, h, aa, bb)
            p_list.append(p.data.argmax())
        return p_list
