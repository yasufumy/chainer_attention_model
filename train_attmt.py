from datetime import datetime

from chainer import optimizers, cuda
from chainer import Variable as V
from chainer import serializers
from tqdm import tqdm

from lib.models import AttentionMT
from lib.preprocessing import gen_lines, line2batch, batch2line
from lib.wrapper import xp
from lib.vocabulary import Vocabulary as vocab
from lib.helper import timer

@timer
def train():
    src_lines = gen_lines('../data/mt/train.de')
    trg_lines = gen_lines('../data/mt/train.en')
    src_vocab = vocab(src_lines, 10)
    trg_vocab = vocab(trg_lines, 10)
    trg_itow = trg_vocab.itow

    attmt = AttentionMT(len(src_vocab), len(trg_vocab), 200, 100)
    attmt.use_gpu(0)
    opt = optimizers.AdaGrad(lr = 0.01)
    opt.setup(attmt)

    n_epoch = 10
    batch_size = 150

    for epoch in tqdm(range(n_epoch)):
        src_lines = gen_lines('../data/mt/train.de')
        trg_lines = gen_lines('../data/mt/train.en')
        src_batches = line2batch(src_lines, src_vocab, batch_size)
        trg_batches = line2batch(trg_lines, trg_vocab, batch_size)
        for x_batch, t_batch in zip(src_batches, trg_batches):
            attmt.zerograds()
            y_batch, loss = attmt(x_batch, t_batch, trg_vocab.wtoi)
            loss.backward()
            opt.update()
            for line in batch2line(y_batch, trg_vocab):
                tqdm.write('epoch: ' + str(epoch) + ' [' + datetime.now().strftime("%Y/%m/%d %H:%M:%S") + ']: ' + line)
    attmt.save_model('test.model')

if __name__ == '__main__':
    try:
        train()
    except KeyboardInterrupt:
        pass
