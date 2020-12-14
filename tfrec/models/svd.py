import tensorflow as tf
from tensorflow import keras


class SVD(keras.Model):
    """
    SVD - A Matrix Factorization Algorithm for Collabarative Filtering as demonstrated by
    Simon Funk in his blogpost - https://sifter.org/~simon/journal/20061211.html

    - Base - keras.Model

    ...

    Attributes
    -----------
    n_users : int
        total number of users in the dataset
    n_items : int
        total number of items in the dataset
    global_mean : float
        mean of ratings in the training set
    embedding_dim : int
        number of factors for user and item (default 50)
    biased : boolean
        whether bias should be used or not (default True)
    init_mean : float
        mean of random initilization for embeddings (default 0)
    init_std_dev : float
        standard deviation of random initilization for embeddings (default 0.1)
    reg_all : float
        l2 regularization factor for all trainable variables (default 0.0001)
    reg_user_embed : float
        l2 regularization factor for user embedding (default reg_all)
    reg_item_embed : float
        l2 regularization factor for item embedding (default reg_all)
    reg_user_bias : float
        l2 regularization factor for user bias (default reg_all)
    reg_item_bias : float
        l2 regularization factor for item bias (default reg_all)
    random_state : integer
        seed variable, useful for reproducing results (default None)
    """

    def __init__(self, n_users, n_items, global_mean, embedding_dim=50, biased=True, init_mean=0, init_std_dev=0.1, reg_all=0.0001,
                 reg_user_embed=None, reg_item_embed=None, reg_user_bias=None, reg_item_bias=None, random_state=None, **kwargs):
        """
        Parameters
        -----------
        n_users : int
            total number of users in the dataset
        n_items : int
            total number of items in the dataset
        global_mean : float
            mean of ratings in the training set
        embedding_dim : int
            number of factors for user and item (default 50)
        biased : boolean
            whether bias should be used or not (default True)
        init_mean : float
            mean of random initilization for embeddings (default 0)
        init_std_dev : float
            standard deviation of random initilization for embeddings (default 0.1)
        reg_all : float
            l2 regularization factor for all trainable variables (default 0.0001)
        reg_user_embed : float
            l2 regularization factor for user embedding (default reg_all)
        reg_item_embed : float
            l2 regularization factor for item embedding (default reg_all)
        reg_user_bias : float
            l2 regularization factor for user bias (default reg_all)
        reg_item_bias : float
            l2 regularization factor for item bias (default reg_all)
        random_state : integer
            seed variable, useful for reproducing results (default None)
        """

        super().__init__(**kwargs)
        self.n_users = n_users
        self.n_items = n_items
        self.global_mean = global_mean
        self.embedding_dim = embedding_dim
        self.biased = biased
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.reg_all = reg_all
        self.reg_user_embed = reg_all if not reg_user_embed else reg_user_embed
        self.reg_item_embed = reg_all if not reg_item_embed else reg_item_embed
        self.reg_user_bias = reg_all if not reg_user_bias else reg_user_bias
        self.reg_item_bias = reg_all if not reg_item_bias else reg_item_bias
        self.random_state = random_state

        self.user_embedding = keras.layers.Embedding(
            input_dim=self.n_users, output_dim=self.embedding_dim,
            embeddings_initializer=tf.keras.initializers.RandomNormal(
                mean=self.init_mean, stddev=self.init_std_dev, seed=self.random_state),
            embeddings_regularizer=tf.keras.regularizers.L2(
                self.reg_user_embed)
        )
        self.item_embedding = keras.layers.Embedding(
            input_dim=self.n_items, output_dim=self.embedding_dim,
            embeddings_initializer=tf.keras.initializers.RandomNormal(
                mean=self.init_mean, stddev=self.init_std_dev, seed=self.random_state),
            embeddings_regularizer=tf.keras.regularizers.L2(
                self.reg_item_embed)
        )
        if self.biased:
            self.user_bias = keras.layers.Embedding(
                input_dim=self.n_users, output_dim=1,
                embeddings_initializer=tf.keras.initializers.Zeros(),
                embeddings_regularizer=tf.keras.regularizers.L2(
                    self.reg_user_bias)
            )
            self.item_bias = keras.layers.Embedding(
                input_dim=self.n_items, output_dim=1,
                embeddings_initializer=tf.keras.initializers.Zeros(),
                embeddings_regularizer=tf.keras.regularizers.L2(
                    self.reg_item_bias)
            )

    def call(self, inputs):
        """Forward pass of input batch"""
        # Separate Inputs
        user, item = inputs[:, 0], inputs[:, 1]

        # Dot Product
        user_embed, item_embed = self.user_embedding(
            user), self.item_embedding(item)
        rating = tf.math.reduce_sum(tf.multiply(
            user_embed, item_embed), 1, keepdims=True)

        # Add global mean and bias if self.bias = True
        if self.biased:
            user_bias, item_bias = self.user_bias(user), self.item_bias(item)
            total_bias = tf.math.add(
                self.global_mean, tf.math.add(user_bias, item_bias))
            rating = tf.math.add(rating, total_bias)

        return rating
