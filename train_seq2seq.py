import os
from datetime import datetime
from argparse import ArgumentParser

from chainer import optimizers, cuda
from chainer import serializers
import numpy as np
from tools.iterator import TextIterator
from tools.text.preprocessing import OneOfMEncoder
from tools.iterable import transpose

from lib.seq2seq import Seq2SeqAttention
from lib.preprocessing import gen_lines
from lib.vocabulary import Vocabulary as vocab
from lib.helper import timer
from lib.config import UNKNOWN_LABEL, START_TOKEN, END_TOKEN

@timer
def train(args):
    src_lines = gen_lines(args.train_src)
    trg_lines = gen_lines(args.train_trg)
    src_vocab = vocab(src_lines, args.unk)
    trg_vocab = vocab(trg_lines, args.unk)
    trg_itow = trg_vocab.itow

    model = Seq2SeqAttention(len(src_vocab), len(trg_vocab), args.embed, args.hidden,
                             trg_vocab.wtoi[START_TOKEN], trg_vocab.wtoi[END_TOKEN])
    model.use_gpu(args.gpu)
    opt = optimizers.AdaGrad(lr=0.01)
    opt.setup(model)

    n_epoch = args.epoch
    batch_size = args.batch

    one_of_m_src = OneOfMEncoder(src_vocab.wtoi, UNKNOWN_LABEL)
    src_train = [one_of_m_src.encode(s) for s in gen_lines(args.train_src)]
    one_of_m_trg = OneOfMEncoder(trg_vocab.wtoi, UNKNOWN_LABEL)
    trg_train = [one_of_m_trg.encode(s) for s in gen_lines(args.train_trg)]

    N = len(src_train)

    for i in range(n_epoch):
        epoch = i + 1
        print('epoch: {}'.format(epoch))
        order = np.random.permutation(N)
        src_batches = TextIterator(src_train, batch_size, order=order)
        trg_batches = TextIterator(trg_train, batch_size, order=order)
        sum_loss = 0
        for x_batch, t_batch in zip(src_batches, trg_batches):
            model.cleargrads()
            loss, y_batch = model.loss(x_batch, t_batch)
            loss.backward()
            opt.update()
            sum_loss += loss.data
            if epoch % 10 == 0:
                for y in transpose(y_batch):
                    print(' '.join(trg_vocab.itow[id_] for id_ in y) + '\n')
        print('loss: {}'.format(sum_loss / N))
    model.save_model(args.model)

    src_test = [one_of_m_src.encode(s) for s in gen_lines(args.test_src)]
    for x_batch in TextIterator(src_test, batch_size, shuffle=False):
        y_hypo = model.inference(x_batch)
        with open(args.output, 'a') as f:
            for h in transpose(y_hypo):
                f.write(' '.join(trg_vocab.itow[id_] for id_ in h) + '\n')

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--embed', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=100)
    parser.add_argument('--batch', type=int, default=120)
    parser.add_argument('--unk', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--train_src')
    parser.add_argument('--train_trg')
    parser.add_argument('--test_src')
    parser.add_argument('--test_trg')
    parser.add_argument('--model', type=str,
                        default=os.path.abspath('model/seq2seq.model'))
    parser.add_argument('--output', type=str,
                        default=os.path.abspath('log/inference.txt'))
    return parser.parse_args()

if __name__ == '__main__':
    try:
        args = parse_args()
        train(args)
    except KeyboardInterrupt:
        pass
