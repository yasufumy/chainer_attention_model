from chainer import functions as F

from wrapper import xp
from config import START_TOKEN, END_TOKEN, IGNORE_LABEL

def gen_lines(filename):
    with open(filename) as f:
        for line in f:
            yield [START_TOKEN] + line.split() + [END_TOKEN]

def line2batch(lines, vocab, batch_size):
    batch = []
    wtoi = vocab.wtoi
    for line in lines:
        batch.append([wtoi[START_TOKEN]]+
                [wtoi[word] for word in line] + [wtoi[END_TOKEN]])
        if len(batch) == batch_size:
            yield F.transpose_sequence(fill_batch(batch))
            batch = []
    if batch:
        yield F.transpose_sequence(fill_batch(batch))

def fill_batch(batch, token=IGNORE_LABEL):
    max_size = max(len(x) for x in batch)
    filled_batch =  xp.array(
                        [x + [token] * (max_size - len(x) + 1) for x in batch],
                        dtype=xp.int32)
    return filled_batch

def batch2line(batches, vocab):
    itow = vocab.itow
    id_list = [batch.data.argmax(axis=1) for batch in batches]
    for i in range(batches[0].data.shape[0]):
        yield ' '.join([itow[int(id_list[k][i])] for k in range(len(id_list))])
