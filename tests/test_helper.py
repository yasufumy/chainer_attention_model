import sys
sys.path.append('..')
import unittest
import io
import re

from helper import timer

class HelperTestCase(unittest.TestCase):

    def test_timer(self):
        import time
        log = io.StringIO()
        sys.stdout = log
        timer(time.sleep)(1)
        elapsed = re.search('\d+', log.getvalue()
                    ).group()
        self.assertEqual('1', elapsed)
