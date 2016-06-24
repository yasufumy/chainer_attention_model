from collections import defaultdict, Counter
from itertools import chain, takewhile

class Vocabulary:
    def __new__(cls, words_generator, th):
        obj = super().__new__(cls)

        word_freq = Counter(chain.from_iterable((words_generator)))

        obj.__stoi = defaultdict(lambda: 0)
        obj.__stoi['<unk>'] = 0
        obj.__stoi['<s>'] = 1
        obj.__stoi['</s>'] = 2
        obj.__itos = []
        obj.__itos.append('<unk>')
        obj.__itos.append('<s>')
        obj.__itos.append('</s>')

        i = 3
        for (k, v) in takewhile(lambda x: x[1] > th, word_freq.most_common()):
            obj.__stoi[k] = i
            obj.__itos.append(k)
            i += 1

        return obj

    def __len__(self):
        return len(self.itos)

    @property
    def stoi(self):
        return self.__stoi

    @property
    def itos(self):
        return self.__itos
