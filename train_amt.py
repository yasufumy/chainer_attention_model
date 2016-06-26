from itertools import tee

from chainer import optimizers, cuda
from chainer import Variable as V

from yasfmy.models import AttentionMT
from yasfmy.preprocessing import gen_lines, gen_batch, gen_word
from yasfmy.wrapper import xp
from yasfmy.vocabulary import Vocabulary as vocab

def main():
    src_lines = gen_lines('../data/mt/train.de')
    trg_lines = gen_lines('../data/mt/train.en')
    train_src_vocab = vocab(src_lines, 10)
    train_trg_vocab = vocab(trg_lines, 10)
    itow = train_trg_vocab.itow

    attmt = AttentionMT(len(train_src_vocab), len(train_trg_vocab), 500, 200)
    attmt.use_gpu(0)
    opt = optimizers.AdaGrad(lr = 0.01)
    opt.setup(attmt)

    n_epoch = 5
    batch_size = 64

    for epoch in range(n_epoch):
        src_lines = gen_lines('../data/mt/train.de')
        trg_lines = gen_lines('../data/mt/train.en')
        src_gen = gen_batch(src_lines, train_src_vocab, batch_size)
        trg_gen = gen_batch(trg_lines, train_trg_vocab, batch_size)
        for x_batch, t_batch in zip(src_gen, trg_gen):
            attmt.zerograds()
            y_batch, loss = attmt(x_batch, t_batch, train_trg_vocab.wtoi)
            word_id_list = [y.argmax(axis=1) for y in y_batch]
            for i in range(len(t_batch)):
                print(' '.join([train_trg_vocab.itow[int(word_id_list[k][i])] for k in range(len(word_id_list))]))
                print()
            loss.backward()
            opt.update()
    serializers.save_npz('test.model', attmt)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
