import chainer
import chainer.functions as F
import chainer.links as L
from tools.model import BaseModel

from config import IGNORE_LABEL, START_TOKEN, END_TOKEN

class Encoder(BaseModel):
    def __init__(self, embed_size, hidden_size):
        super().__init__(
            # input weight vector of {input, output, forget} gate and input
            W = L.Linear(embed_size, 4 * hidden_size),
            # hidden weight vector of {input, output, forget} gate and input
            U = L.Linear(hidden_size, 4 * hidden_size),
        )
        self.hidden_size = hidden_size

    def __call__(self, embeded_x, m_prev, h_prev, enable):
        batch_size = embeded_x.shape[0]
        lstm_in = self.W(embeded_x) + self.U(h_prev)
        m_tmp, h_tmp = F.lstm(m_prev, lstm_in)
        # flags if feeding previous output
        enable = F.broadcast_to(F.expand_dims(enable, -1),
                                (batch_size, self.hidden_size))
        m = F.where(enable, m_tmp, m_prev)
        h = F.where(enable, h_tmp, h_prev)
        return m, h

class AttentionDecoder(BaseModel):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__(
            # Weights of Decoder
            E = L.EmbedID(vocab_size, embed_size, IGNORE_LABEL),
            W = L.Linear(embed_size, 4 * hidden_size),
            U = L.Linear(hidden_size, 4 * hidden_size),
            C = L.Linear(2 * hidden_size, 4 * hidden_size),
            U_o = L.Linear(hidden_size, hidden_size),
            V_o = L.Linear(embed_size, hidden_size),
            C_o = L.Linear(2 * hidden_size, hidden_size),
            W_o = L.Linear(hidden_size, vocab_size),
            # Weights of Attention
            U_a = L.Linear(2 * hidden_size, hidden_size),
            W_a = L.Linear(hidden_size, hidden_size),
            v_a = L.Linear(hidden_size, 1),
        )
        self.hidden_size = hidden_size

    def _attention(self, h_forward, h_backword, s):
        batch_size = s.shape[0]
        sentence_size = len(h_forward)
        hidden_size = self.hidden_size

        weighted_s = F.broadcast_to(F.expand_dims(self.W_a(s), axis=1),
                                    (batch_size, sentence_size, hidden_size))
        h = F.concat((F.concat(h_forward, axis=0), F.concat(h_backword, axis=0)))
        weighted_h = F.reshape(self.U_a(h), (batch_size, sentence_size, hidden_size))

        e = self.v_a(F.reshape(F.tanh(weighted_s + weighted_h),
                               (batch_size * sentence_size, hidden_size)))
        e = F.reshape(e, (batch_size, sentence_size))
        alpha = F.softmax(e)
        c = F.batch_matmul(F.reshape(h, (batch_size, 2 * hidden_size, sentence_size)), alpha)
        return F.reshape(c, (batch_size, 2 * hidden_size))

    def __call__(self, y, m_prev, s_prev, h_forward, h_backword):
        # m is memory cell of lstm, s is previous hidden output
        # calculate attention
        c = self._attention(h_forward, h_backword, s_prev)
        # decode once
        embeded_y = self.E(y)
        batch_size = y.shape[0]
        lstm_in = self.W(embeded_y) + self.U(s_prev) + self.C(c)
        m_tmp, s_tmp = F.lstm(m_prev, lstm_in)
        enable = F.broadcast_to(F.expand_dims(y.data != IGNORE_LABEL, -1),
                                (batch_size, self.hidden_size))
        m = F.where(enable, m_tmp, m_prev)
        s = F.where(enable, s_tmp, s_prev)
        t = self.U_o(s) + self.V_o(embeded_y) + self.C_o(c)
        return self.W_o(t), m, s

class Seq2SeqAttention(BaseModel):
    def __init__(self, src_size, trg_size, embed_size, hidden_size,
                 start_token_id, end_token_id):
        super().__init__(
            embed = L.EmbedID(src_size, embed_size, IGNORE_LABEL),
            f_encoder = Encoder(embed_size, hidden_size),
            b_encoder = Encoder(embed_size, hidden_size),
            W_s = L.Linear(hidden_size, hidden_size),
            decoder = AttentionDecoder(trg_size, embed_size, hidden_size)
        )
        self.hidden_size = hidden_size
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id

    def loss(self, src, trg):
        # preparing
        batch_size = src[0].shape[0]
        self.prepare(batch_size)
        # encoding
        h_forward, h_backword = self.encode(src)
        # decoding with attention
        loss, y_batch = self.decode_train(trg, h_forward, h_backword)
        return loss, y_batch

    def inference(self, src, limit=20):
        # preparing
        batch_size = src[0].shape[0]
        self.prepare(batch_size)
        # encoding
        h_forward, h_backword = self.encode(src)
        # decoding with attention
        y_hypo = self.decode_inference(h_forward, h_backword, limit)
        return y_hypo

    def encode(self, src):
        fm = fh = bm = bh = self.initial_state
        h_forward = []
        h_backword = []
        for fx, bx in zip(src, src[::-1]):
            f_enable = (fx.data != IGNORE_LABEL)
            b_enable = (bx.data != IGNORE_LABEL)
            embeded_fx = self.embed(fx)
            embeded_bx = self.embed(bx)
            fm, fh = self.f_encoder(embeded_fx, fm, fh, f_enable)
            bm, bh = self.b_encoder(embeded_bx, bm, bh, b_enable)
            h_forward.append(fh)
            h_backword.insert(0, bh)
        return h_forward, h_backword

    def decode_train(self, trg, h_forward, h_backword):
        m = self.initial_state
        s = F.tanh(self.W_s(h_backword[0]))
        y = self.initial_y
        y_batch = []
        loss = 0
        for t in trg:
            y, m, s = self.decoder(y, m, s, h_forward, h_backword)
            y_batch.append(y.data.argmax(1).tolist())
            loss += F.softmax_cross_entropy(y, t)
            y = t
        return loss, y_batch

    def decode_inference(self, h_forward, h_backword, limit=20):
        m = self.initial_state
        s = F.tanh(self.W_s(h_backword[0]))
        y = self.initial_y
        y_hypo = []
        end_token_id = self.end_token_id
        xp = self.xp
        for _ in range(limit):
            y, m, s = self.decoder(y, m, s, h_forward, h_backword)
            p = y.data.argmax(1)
            if all(p == end_token_id):
                break
            y_hypo.append(p.tolist())
            y = chainer.Variable(p.astype(xp.int32))
        return y_hypo

    def prepare(self, batch_size):
        xp = self.xp
        self.initial_state = chainer.Variable(xp.zeros(
                            (batch_size, self.hidden_size), dtype=xp.float32))
        self.initial_y = chainer.Variable(xp.array(
                            [self.start_token_id] * batch_size, dtype=xp.int32))
