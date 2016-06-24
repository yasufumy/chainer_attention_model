from itertools import chain
import io

from wrapper import xp
from wrapper import wrap_variable

def gen_lines(filename):
    with open(filename) as f:
        for line in f:
            yield line.split()

def gen_batch(lines, vocab, batch_size):
    batch = []
    wtoi = vocab.wtoi
    for line in lines:
        batch.append([wtoi['<s>']]+
                [wtoi[word] for word in line] + [wtoi['</s>']])
        if len(batch) == batch_size:
            yield fill_batch(batch)
            batch = []
    if batch:
        yield batch

@wrap_variable
def gen_word(batch):
    batch_len = len(batch)
    line_len = len(batch[0])
    for l in range(line_len):
        yield xp.array([batch[k][l] for k in range(batch_len)],
                        dtype=xp.int32)

def fill_batch(batch, token=-1):
    max_len = max(len(x) for x in batch)
    return [x + [token] * (max_len - len(x) + 1) for x in batch]
