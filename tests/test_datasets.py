import unittest
import pandas as pd
from tfrec.datasets import fetch_ml_100k


class Test_ml_100k(unittest.TestCase):
    def test_instance(self):
        dataset = fetch_ml_100k()
        self.assertIsInstance(dataset, int)
