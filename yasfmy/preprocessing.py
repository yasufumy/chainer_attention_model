from itertools import chain
import io

class Preprocessing:
    @classmethod
    def get_vocab(cls, document):
        vocab = {}
        if type(document) is io.TextIOWrapper:
            words = document.read().rstrip('\n').split()
        elif type(document) is str:
            words = document.rstrip('\n').split()
        elif type(document) is list:
            if cls._is_nested(document):
                words = cls.flatten(document)
            else:
                words = document
        for word in words:
            if word not in vocab:
                vocab[word] = len(vocab)
        return vocab

    @staticmethod
    def flatten(nested):
        return list(chain.from_iterable(nested))

    @staticmethod
    def _is_nested(arr):
        return any(isinstance(i, list) for i in arr)
