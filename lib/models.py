import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import serializers

from wrapper import xp
from config import IGNORE_LABEL, START_TOKEN

class BaseModel(chainer.Chain):
    def __call__(self):
        pass

    def use_gpu(self, gpu_id):
        cuda.get_device(gpu_id).use()
        self.to_gpu()

    def save_model(self, filename):
        self.to_cpu()
        serializers.save_npz(filename, self)

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
            abw = L.Linear(2 * hidden_size, hidden_size),
            # previous hidden output weight vector
            pw = L.Linear(hidden_size, hidden_size),
            # attention output weight vector
            we = L.Linear(hidden_size, 1),
        )
        self.hidden_size = hidden_size

    def __call__(self, a_list, b_list, p):
        #batch_size = p.data.shape[0]
        #e_list = []
        #sum_e = xp.Zeros((batch_size, 1), dtype=xp.float32)
        #for a, b in zip(a_list, b_list):
        #    w = F.tanh(self.aw(a) + self.bw(b) + self.pw(p))
        #    e = F.exp(self.we(w))
        #    e_list.append(e)
        #    sum_e += e
        #aa = bb = xp.Zeros((batch_size, self.hidden_size), dtype=xp.float32)
        #for a, b, e in zip(a_list, b_list, e_list):
        #    e /= sum_e
        #    aa += F.reshape(F.batch_matmul(a, e), (batch_size, self.hidden_size))
        #    bb += F.reshape(F.batch_matmul(b, e), (batch_size, self.hidden_size))
        batch_size = p.data.shape[0]
        sent_size = len(a_list)
        wp = F.expand_dims(self.pw(p), axis=1)
        wp = F.broadcast_to(wp, (batch_size, sent_size, self.hidden_size))
        wp = F.concat((wp, wp), axis=2)
        a = F.concat(a_list, axis=0)
        b = F.concat(b_list, axis=0)
        hab = F.concat((a, b))
        wab = self.abw(hab)
        wab = F.reshape(wab, (batch_size, sent_size, self.hidden_size))
        e = self.we(F.reshape(F.tanh(wab), (batch_size * sent_size, self.hidden_size)))
        e = F.reshape(e, (batch_size, sent_size))
        att = F.softmax(e)
        ab = F.batch_matmul(F.reshape(hab, (batch_size, 2 * self.hidden_size, sent_size)), att)
        return F.reshape(ab, (batch_size, 2 * self.hidden_size))

class AttentionMT(BaseModel):
    def __init__(self, src_size, trg_size, embed_size, hidden_size):
        super().__init__(
            emb = L.EmbedID(src_size, embed_size, IGNORE_LABEL),
            fenc = Encoder(embed_size, hidden_size),
            benc = Encoder(embed_size, hidden_size),
            att = Attention(hidden_size),
            dec = Decoder(trg_size, embed_size, hidden_size)
        )
        self.embed_size = embed_size
        self.hidden_size = hidden_size

    def __call__(self, src, trg, trg_wtoi):
        # preparing
        batch_len = src[0].data.shape[0]
        hidden_init = xp.Zeros((batch_len, self.hidden_size), dtype=xp.float32)
        y = xp.Array([trg_wtoi[START_TOKEN] for _ in range(batch_len)], dtype=xp.int32)
        # embeding words
        x_list = [F.tanh(self.emb(x)) for x in src]
        # encoding
        a_list, b_list = self.forward_enc(x_list, hidden_init)
        # attention
        y_batch, loss = self.forward_dec_train(trg, a_list, b_list, (hidden_init, y))
        return y_batch, loss

    def forward_enc(self, x_list, initial_value):
        fc = fh = bc = bh = initial_value
        fenc_list = benc_list = []
        for fx, bx in zip(x_list, reversed(x_list)):
            fc, fh = self.fenc(fx, fc, fh)
            bc, bh = self.benc(bx, bc, bh)
            fenc_list.append(fh)
            benc_list.append(bh)
        return fenc_list, benc_list

    def forward_dec_train(self, trg, a_list, b_list, initial_value):
        h = c = initial_value[0]
        y = initial_value[1]
        y_batch = []
        loss = xp.Zeros(None, dtype=xp.float32)
        for t in trg:
            ab = self.att(a_list, b_list, h)
            y, c, h = self.dec(y, c, h, ab)
            y_batch.append(y)
            loss += F.softmax_cross_entropy(y, t)
            y = t
        return y_batch, loss
