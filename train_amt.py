from datetime import datetime

from chainer import optimizers, cuda
from chainer import Variable as V
from chainer import serializers
from tqdm import tqdm

from yasfmy.models import AttentionMT
from yasfmy.preprocessing import gen_lines, gen_batch, gen_word
from yasfmy.wrapper import xp
from yasfmy.vocabulary import Vocabulary as vocab
from yasfmy.helper import timer

@timer
def train():
    src_lines = gen_lines('../data/mt/train.de')
    trg_lines = gen_lines('../data/mt/train.en')
    src_vocab = vocab(src_lines, 10)
    trg_vocab = vocab(trg_lines, 10)
    itow = trg_vocab.itow

    attmt = AttentionMT(len(src_vocab), len(trg_vocab), 200, 100)
    attmt.use_gpu(0)
    opt = optimizers.AdaGrad(lr = 0.01)
    opt.setup(attmt)

    n_epoch = 10
    batch_size = 200

    for epoch in tqdm(range(n_epoch)):
        src_lines = gen_lines('../data/mt/train.de')
        trg_lines = gen_lines('../data/mt/train.en')
        src_gen = gen_batch(src_lines, src_vocab, batch_size)
        trg_gen = gen_batch(trg_lines, trg_vocab, batch_size)
        for x_batch, t_batch in zip(src_gen, trg_gen):
            attmt.zerograds()
            y_batch, loss = attmt(x_batch, t_batch, trg_vocab.wtoi)
            word_id_list = [y.argmax(axis=1) for y in y_batch]
            for i in range(len(t_batch)):
                tqdm.write('epoch: ' + str(epoch) + ' [' + datetime.now().strftime("%Y/%m/%d %H:%M:%S") + ']: ' +\
                        ' '.join([trg_vocab.itow[int(word_id_list[k][i])] for k in range(len(word_id_list))]))
            loss.backward()
            opt.update()
    attmt.save_model('test.model')

if __name__ == '__main__':
    try:
        train()
    except KeyboardInterrupt:
        pass
