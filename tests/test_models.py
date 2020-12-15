import unittest
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tfrec.models import SVD, SVDpp
from tfrec.datasets import fetch_ml_100k
from tfrec.utils import preprocess_and_split

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
data = fetch_ml_100k()
dataset, user_item_encodings = preprocess_and_split(data, random_state=seed)


class TestModels(unittest.TestCase):
    def setUp(self):
        # Answer to the Ultimate Question of Life, the Universe, and Everything
        self.seed = 42

        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        self.data = fetch_ml_100k()
        self.dataset, user_item_encodings = preprocess_and_split(data, random_state=self.seed)

        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.dataset
        self.num_users = len(np.unique(data['userId']))
        self.num_movies = len(np.unique(data['movieId']))
        self.global_mean = np.mean(data['rating'])

        self.batch_size = 128
        self.learning_rate = 0.007
        self.epochs = 1
        self.opt = keras.optimizers.Adam(self.learning_rate)
        self.loss_fn = keras.losses.MeanSquaredError()

    def test_svd(self):
        """Test Custom Model SVD to avoid breaking in future tensorflow versions"""

        model = SVD(self.num_users, self.num_movies, self.global_mean, random_state=self.seed)
        model.compile(loss=self.loss_fn, optimizer=self.opt)
        history = model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs)
        # Obvious Test
        self.assertIsInstance(history.model, SVD)

    def test_svdpp(self):
        """Test Custom Model SVD++ to avoid breaking in future tensorflow versions"""

        model = SVDpp(self.num_users, self.num_movies, self.global_mean, random_state=self.seed)
        model.implicit_feedback(self.x_train)
        model.compile(loss=self.loss_fn, optimizer=self.opt)
        history = model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs)
        # Obvious Test
        self.assertIsInstance(history.model, SVDpp)
