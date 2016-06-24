from collections import defaultdict, Counter
from itertools import chain, takewhile

class Vocabulary:
    def __new__(cls, words_generator, th):
        obj = super().__new__(cls)

        word_freq = Counter(chain.from_iterable((words_generator)))

        obj.__wtoi = defaultdict(lambda: 0)
        obj.__wtoi['<unk>'] = 0
        obj.__wtoi['<s>'] = 1
        obj.__wtoi['</s>'] = 2
        obj.__itow = []
        obj.__itow.append('<unk>')
        obj.__itow.append('<s>')
        obj.__itow.append('</s>')

        i = 3
        for (k, v) in takewhile(lambda x: x[1] > th, word_freq.most_common()):
            obj.__wtoi[k] = i
            obj.__itow.append(k)
            i += 1

        return obj

    def __len__(self):
        return len(self.itow)

    @property
    def wtoi(self):
        return self.__wtoi

    @property
    def wtos(self):
        return self.__itow
