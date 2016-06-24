from itertools import tee

from chainer import optimizers, cuda
from chainer import Variable as V

from yasfmy.models import AttentionMT
from yasfmy.preprocessing import gen_lines, gen_batch, gen_word
from yasfmy.wrapper import xp
from yasfmy.vocabulary import Vocabulary as vocab

def main():
    src_lines1, src_lines2 = tee(gen_lines('../data/mt/train.de'), 2)
    trg_lines1, trg_lines2 = tee(gen_lines('../data/mt/train.en'), 2)
    train_src_vocab = vocab(src_lines1, 10)
    train_trg_vocab = vocab(trg_lines1, 10)
    itow = train_trg_vocab.itos

    attmt = AttentionMT(len(train_src_vocab), len(train_trg_vocab), 500, 200)
    cuda.get_device(0).use()
    attmt.to_gpu()
    opt = optimizers.AdaGrad(lr = 0.01)
    opt.setup(attmt)

    n_epoch = 5
    batch_size = 64

    for epoch in range(n_epoch):
        src_gen = gen_batch(src_lines2, train_src_vocab, batch_size)
        trg_gen = gen_batch(trg_lines2, train_trg_vocab, batch_size)
        for x_batch, t_batch in zip(src_gen, trg_gen):
            attmt.zerograds()
            y_batch, loss = attmt(x_batch, t_batch, train_trg_vocab.wtoi)
            loss.backward()
            opt.update()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
