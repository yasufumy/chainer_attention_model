from collections import defaultdict, Counter
from itertools import chain, takewhile

from config import START_TOKEN, END_TOKEN, UNKNOWN_TOKEN

class Vocabulary:
    def __new__(cls, words_generator, th):
        obj = super().__new__(cls)

        word_freq = Counter(chain.from_iterable((words_generator)))

        obj.__wtoi = defaultdict(lambda: 0)
        obj.__wtoi[UNKNOWN_TOKEN] = 0
        obj.__wtoi[START_TOKEN] = 1
        obj.__wtoi[END_TOKEN] = 2
        obj.__itow = []
        obj.__itow.append(UNKNOWN_TOKEN)
        obj.__itow.append(START_TOKEN)
        obj.__itow.append(END_TOKEN)

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
    def itow(self):
        return self.__itow
