import sys
sys.path.append('..')
import unittest
import io
import re

from preprocessing import gen_lines, gen_batch, fill_batch
from vocabulary import Vocabulary

class PreprocessingTestCase(unittest.TestCase):
    def test_gen_lines(self):
        pass

    def test_gen_batch(self):
        lines = ['a', 'b', 'c']
        vocab = Vocabulary(lines, th=0)
        batches = gen_batch(lines, vocab, 3)
        lines = [[[1] + [vocab.wtoi[l]] + [2, -1]] for l in lines]
        for batch, line in zip(batches, lines):
            for b, l in zip(batch, line):
                self.assertEqual(b, l)

    def test_fill_batch(self):
        lines= [[1, 2, 3], [1, 2]]
        batch = fill_batch(lines)
        self.assertEqual(batch[1][-1], -1)
