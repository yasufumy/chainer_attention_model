from chainer import optimizers, cuda
from chainer import Variable as V
from tqdm import tqdm
import numpy as np

from yasfmy.models import AttentionMT
from yasfmy.preprocessing import gen_lines, gen_batch, gen_word
from yasfmy.wrapper import xp
from yasfmy.vocabulary import Vocabulary as vocab

def main():
    train_src_vocab = vocab(gen_lines('../data/mt/train.de'), 10)
    train_trg_vocab = vocab(gen_lines('../data/mt/train.en'), 10)

    attmt = AttentionMT(len(train_src_vocab), len(train_trg_vocab), 500, 200)
    cuda.get_device(0).use()
    attmt.to_gpu()
    opt = optimizers.AdaGrad(lr = 0.01)
    opt.setup(attmt)

    n_epoch = 5
    batch_size = 64

    for epoch in tqdm(range(n_epoch), desc='epoch loop', leave=False):
        train_src_lines = gen_lines('../data/mt/train.de')
        train_trg_lines = gen_lines('../data/mt/train.en')
        src_gen = gen_batch(train_src_lines, train_src_vocab, batch_size)
        trg_gen = gen_batch(train_trg_lines, train_trg_vocab, batch_size)
        for x_batch, t_batch in zip(src_gen, trg_gen):
            attmt.zerograds()
            y_batch, loss = attmt(x_batch, t_batch, train_trg_vocab.stoi)
            loss.backward()
            opt.update()
            tqdm.write('loss:%f' % loss.data)

if __name__ == '__main__':
    main()
