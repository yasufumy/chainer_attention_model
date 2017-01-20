from collections import defaultdict, Counter
from itertools import chain, takewhile

from config import START_TOKEN, END_TOKEN, UNKNOWN_TOKEN, UNKNOWN_LABEL

class Vocabulary:
    def __new__(cls, words_generator, th=5):
        obj = super().__new__(cls)

        word_freq = Counter(chain.from_iterable((words_generator)))

        obj.wtoi = defaultdict(lambda: 0)
        obj.wtoi[UNKNOWN_TOKEN] = UNKNOWN_LABEL
        obj.wtoi[START_TOKEN] = 1
        obj.wtoi[END_TOKEN] = 2
        obj.itow = []
        obj.itow.append(UNKNOWN_TOKEN)
        obj.itow.append(START_TOKEN)
        obj.itow.append(END_TOKEN)

        i = 3
        for (k, v) in takewhile(lambda x: x[1] > th, word_freq.most_common()):
            obj.wtoi[k] = i
            obj.itow.append(k)
            i += 1

        return obj

    def __len__(self):
        return len(self.itow)
