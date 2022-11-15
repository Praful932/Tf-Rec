import unittest
import numpy as np
from tfrec.datasets import fetch_ml_100k
from tfrec.utils.model_utils import preprocess_and_split


class TestPreprocess(unittest.TestCase):
    def setUp(self):
        self.dataset = fetch_ml_100k()

    def test_preprocess_instance(self):
        preprocessed_dataset, user_item_encodings = preprocess_and_split(self.dataset)

        (x_train, y_train), (x_test, y_test) = preprocessed_dataset

        self.assertIsInstance(x_train, np.ndarray)
        self.assertIsInstance(y_train, np.ndarray)
        self.assertIsInstance(x_test, np.ndarray)
        self.assertIsInstance(y_test, np.ndarray)

        self.assertEqual(len(x_train) + len(x_test), len(self.dataset))

        (
            user_to_encoded,
            encoded_to_user,
            item_to_encoded,
            encoded_to_item,
        ) = user_item_encodings

        for instance in user_item_encodings:
            self.assertIsInstance(instance, dict)

        self.assertEqual(len(user_to_encoded), len(encoded_to_user))
        self.assertEqual(len(item_to_encoded), len(encoded_to_item))
