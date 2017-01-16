import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import serializers

from wrapper import xp
from config import IGNORE_LABEL, START_TOKEN, END_TOKEN

INF = float('inf')

class BaseModel(chainer.Chain):
    def __call__(self):
        pass

    def use_gpu(self, gpu_id):
        cuda.get_device(gpu_id).use()
        self.to_gpu()

    def save_model(self, filename):
        self.to_cpu()
        serializers.save_hdf5(filename, self)

    def load_model(self, filename):
        serializers.load_hdf5(filename, self)

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
            ye = L.EmbedID(vocab_size, embed_size, IGNORE_LABEL),
            # input(previous output) weight vector
            eh = L.Linear(embed_size, 4 * hidden_size),
            # hidden weight vector
            hh = L.Linear(hidden_size, 4 * hidden_size),
            # forward encoder weight vector
            abh = L.Linear(2 * hidden_size, 4 * hidden_size),
            # decoder weight weight vector
            hf = L.Linear(hidden_size, embed_size),
            # output weight vector
            fy = L.Linear(embed_size, vocab_size)
        )

    def __call__(self, y, c, h, ab):
        e = F.tanh(self.ye(y))
        c, h = F.lstm(c, self.eh(e) + self.hh(h) + self.abh(ab))
        f = F.tanh(self.hf(h))
        return self.fy(f), c, h

class Attention(BaseModel):
    def __init__(self, hidden_size):
        super().__init__(
            # forward + backward encoder weight vector
            U_a = L.Linear(2 * hidden_size, hidden_size),
            # previous hidden output weight vector
            W_a = L.Linear(hidden_size, hidden_size),
            # attention output weight vector
            v_a = L.Linear(hidden_size, 1),
        )
        self.hidden_size = hidden_size

    def __call__(self, a_list, b_list, s):
        # preparing
        batch_size = s.data.shape[0]
        sent_size = len(a_list)
        # expand pw(p) to size of a matrix of encoder outputs
        weighted_s = F.expand_dims(self.W_a(s), axis=1)
        weighted_s = F.broadcast_to(weighted_s, (batch_size, sent_size, self.hidden_size))
        # make a matrix of encoder + decoder (batch_size * sent_size, 2 * hidden_size)
        h = F.concat((F.concat(a_list, axis=0), F.concat(b_list, axis=0)))
        # apply a weight vector to a encoder matrix
        weighted_h = self.U_a(h)
        weighted_h = F.reshape(weighted_h, (batch_size, sent_size, self.hidden_size))
        weighted_h = F.where(weighted_h.data!=0, weighted_h,
                             xp.ones(weighted_h.data.shape, dtype=xp.float32)*-INF)
        # apply a weight vector after calculating elementwise addtion (enc matrix + prev hidden output)
        e = self.v_a(F.reshape(F.tanh(weighted_s + weighted_h),
                   (batch_size * sent_size, self.hidden_size)))
        e = F.reshape(e, (batch_size, sent_size))
        # weights how amount enc vector affects
        alpha = F.softmax(e)
        c = F.batch_matmul(F.reshape(h, (batch_size, 2 * self.hidden_size, sent_size)), alpha)
        return F.reshape(c, (batch_size, 2 * self.hidden_size))

class AttentionDecoder(BaseModel):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__(
            # Weights of Decoder
            E = L.EmbedID(vocab_size, embed_size, IGNORE_LABEL),
            W = L.Linear(embed_size, 4 * hidden_size),
            U = L.Linear(hidden_size, 4 * hidden_size),
            C = L.Linear(2 * hidden_size, 4 * hidden_size),
            W_o = L.Linear(hidden_size, vocab_size),
            # Weights of Attention
            U_a = L.Linear(2 * hidden_size, hidden_size),
            W_a = L.Linear(hidden_size, hidden_size),
            v_a = L.Linear(hidden_size, 1),
        )
        self.hidden_size = hidden_size

    def _attention(self, h_forward, h_backword, s):
        batch_size = s.data.shape[0]
        sentence_size = len(h_forward)

        weighted_s = F.expand_dims(self.W_a(s), axis=1)
        weighted_s = F.broadcast_to(weighted_s,
                                    (batch_size, sentence_size, self.hidden_size))
        h = F.concat((F.concat(h_forward, axis=0), F.concat(h_backword, axis=0)))
        weighted_h = self.U_a(h)
        weighted_h = F.reshape(weighted_h, (batch_size, sentence_size, self.hidden_size))
        weighted_h = F.where(weighted_h.data!=0, weighted_h,
                             self.xp.ones(weighted_h.data.shape, dtype=self.xp.float32)*-INF)

        e = self.v_a(F.reshape(F.tanh(weighted_s + weighted_h),
                               (batch_size * sentence_size, self.hidden_size)))
        e = F.reshape(e, (batch_size, sentence_size))
        alpha = F.softmax(e)
        c = F.batch_matmul(F.reshape(h, (batch_size, 2 * self.hidden_size, sentence_size)), alpha)
        return F.reshape(c, (batch_size, 2 * self.hidden_size))

    def __call__(self, y, m, s, h_forward, h_backword):
        # m is memory cell of lstm, s is previous hidden output
        # calculate attention
        c = self._attention(h_forward, h_backword, s)
        # decode once
        m, s = F.lstm(m, self.W(F.tanh(self.E(y))) + self.U(s) + self.C(c))
        return self.W_o(s), m, s

class AttentionMT(BaseModel):
    def __init__(self, src_size, trg_size, embed_size, hidden_size):
        super().__init__(
            emb = L.EmbedID(src_size, embed_size, IGNORE_LABEL),
            fenc = Encoder(embed_size, hidden_size),
            benc = Encoder(embed_size, hidden_size),
            dec = AttentionDecoder(trg_size, embed_size, hidden_size)
        )
        self.hidden_size = hidden_size

    def __call__(self, src, trg, trg_wtoi):
        # preparing
        batch_size = src[0].data.shape[0]
        self.hidden_init = xp.Zeros((batch_size, self.hidden_size), dtype=xp.float32)
        y = xp.Array([trg_wtoi[START_TOKEN] for _ in range(batch_size)], dtype=xp.int32)
        # embeding words
        x_list = [F.tanh(self.emb(x)) for x in src]
        # encoding
        a_list, b_list = self.forward_enc(x_list)
        # attention
        y_batch, loss = self.forward_dec_train(trg, a_list, b_list, y)
        return y_batch, loss

    def forward_enc(self, x_list):
        fc = fh = bc = bh = self.hidden_init
        fenc_list = benc_list = []
        for fx, bx in zip(x_list, x_list[::-1]):
            fc, fh = self.fenc(fx, fc, fh)
            bc, bh = self.benc(bx, bc, bh)
            fenc_list.append(fh)
            benc_list.append(bh)
        return fenc_list, benc_list

    def forward_dec_train(self, trg, a_list, b_list, y):
        h = c = self.hidden_init
        y_batch = []
        loss = xp.Array(0, dtype=xp.float32)
        for t in trg:
            y, c, h = self.dec(y, c, h, a_list, b_list)
            y_batch.append(y)
            loss += F.softmax_cross_entropy(y, t)
            y = t
        return y_batch, loss

    def test(self, src, trg, limit=20):
        # preparing
        batch_size = src[0].data.shape[0]
        trg_wtoi = trg.wtoi
        self.hidden_init = xp.Zeros((batch_size, self.hidden_size), dtype=xp.float32)
        y = xp.Array([trg_wtoi[START_TOKEN] for _ in range(batch_size)], dtype=xp.int32)
        # embeding words
        x_list = [F.tanh(self.emb(x)) for x in src]
        a_list, b_list = self.forward_enc(x_list)
        h = c = self.hidden_init
        y_line = []
        for _ in range(limit):
            ab = self.att(a_list, b_list, h)
            y, c, h = self.dec(y, c, h, ab)
            w = trg.itow[int(y.data.argmax(axis=1))]
            if w == END_TOKEN:
                break
            y_line.append(w)
            y = xp.Array(y.data.argmax(axis=1), dtype=xp.int32)
        return y_line
