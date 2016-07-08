from datetime import datetime
from argparse import ArgumentParser

from chainer import optimizers, cuda
from chainer import serializers

from lib.models import AttentionMT
from lib.preprocessing import gen_lines, line2batch, batch2line
from lib.wrapper import xp
from lib.vocabulary import Vocabulary as vocab
from lib.helper import timer

@timer
def train(args):
    src_lines = gen_lines(args.train_src)
    trg_lines = gen_lines(args.train_trg)
    src_vocab = vocab(src_lines, args.unk)
    trg_vocab = vocab(trg_lines, args.unk)
    trg_itow = trg_vocab.itow

    attmt = AttentionMT(len(src_vocab), len(trg_vocab), args.embed, args.hidden)
    attmt.use_gpu(0)
    opt = optimizers.AdaGrad(lr=0.01)
    opt.setup(attmt)

    n_epoch = args.epoch
    batch_size = args.batch

    for epoch in range(n_epoch):
        src_lines = gen_lines(args.train_src)
        trg_lines = gen_lines(args.train_trg)
        src_batches = line2batch(src_lines, src_vocab, batch_size)
        trg_batches = line2batch(trg_lines, trg_vocab, batch_size)
        for x_batch, t_batch in zip(src_batches, trg_batches):
            attmt.zerograds()
            y_batch, loss = attmt(x_batch, t_batch, trg_vocab.wtoi)
            loss.backward()
            opt.update()
            for line in batch2line(y_batch, trg_vocab):
                print('epoch: ' + str(epoch + 1) +
                        ' [' + datetime.now().strftime("%Y/%m/%d %H:%M:%S") +
                        ']: ' + line)
    attmt.save_model('test.model')
    return attmt

def test(attmt, args):
    src_lines = gen_lines(args.train_src)
    trg_lines = gen_lines(args.train_trg)
    src_vocab = vocab(src_lines, args.unk)
    trg_vocab = vocab(trg_lines, args.unk)
    attmt.use_gpu(args.gpu)
    for x_batch in line2batch(gen_lines(args.test_src), src_vocab, 1):
        y_batch = attmt.test(x_batch, trg_vocab)
        with open(args.output, 'a') as f:
            f.write(' '.join(y_batch).replace('<s> ', '').replace(' </s>',  '') + '\n')

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
    parser.add_argument('--model')
    parser.add_argument('--output')
    return parser.parse_args()

if __name__ == '__main__':
    try:
        args = parse_args()
        attmt = train(args)
        test(attmt, args)
    except KeyboardInterrupt:
        pass
