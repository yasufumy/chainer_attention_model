import prepare
import unittest
from vocabulary import Vocabulary

class VocabularyTestCode(unittest.TestCase):
    def test_new(self):
        words = ['a', 'b', 'c', 'a', 'a']
        vocab = Vocabulary(words, th=2)
        self.assertEqual(len(vocab), 4)
        self.assertIn('a', vocab.wtoi)
