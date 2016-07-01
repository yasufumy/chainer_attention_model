from itertools import chain
import io

from chainer import Variable as V

from wrapper import xp

def gen_lines(filename):
    with open(filename) as f:
        for line in f:
            yield line.split()

def line2batch(lines, vocab, batch_size):
    batch = []
    wtoi = vocab.wtoi
    for line in lines:
        batch.append([wtoi['<s>']]+
                [wtoi[word] for word in line] + [wtoi['</s>']])
        if len(batch) == batch_size:
            #yield fill_batch(batch)
            yield [V(word) for word in fill_batch(batch).T]
            batch = []
    if batch:
        yield [V(word) for word in fill_batch(batch).T]

def fill_batch(batch, token=-1):
    max_len = max(len(x) for x in batch)
    return xp.array([x + [token] * (max_len - len(x) + 1) for x in batch], dtype=xp.int32)

def batch2line(batches, vocab):
    itow = vocab.itow
    id_list = [batch.data.argmax(axis=1) for batch in batches]
    for i in range(len(batches)):
        yield ' '.join([itow[int(id_list[k][i])] for k in range(len(id_list))])
