import tensorflow as tf
from tensorflow import keras


class SVDpp(keras.Model):
    """
    SVD++ - An extension of the SVD Model employing implicit Feedback as originally
    demonstrated in the paper "Factorization meets the neighborhood: a multifaceted collaborative filtering model".

    - Base - keras.Model

    ...

    Attributes:
    -----------
    n_users : int
        total number of users in the dataset
    n_items : int
        total number of items in the dataset
    global_mean : float
        mean of ratings in the training set
    embedding_dim : int
        number of factors for user and item (default 50)
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

    def __init__(self, n_users, n_items, global_mean, embedding_dim=50, init_mean=0, init_std_dev=0.1, reg_all=0.0001,
                 reg_user_embed=None, reg_item_embed=None, reg_impl_embed=None, reg_user_bias=None, reg_item_bias=None, random_state=None, **kwargs):
        super().__init__(**kwargs)
        self.n_users = n_users
        self.n_items = n_items
        self.global_mean = global_mean
        self.embedding_dim = embedding_dim
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.reg_all = reg_all
        self.reg_user_embed = reg_all if not reg_user_embed else reg_user_embed
        self.reg_item_embed = reg_all if not reg_item_embed else reg_item_embed
        self.reg_impl_embed = reg_all if not reg_impl_embed else reg_impl_embed
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
        self.item_implicit_embedding = keras.layers.Embedding(
            input_dim=self.n_items, output_dim=self.embedding_dim,
            embeddings_initializer=tf.keras.initializers.RandomNormal(
                mean=self.init_mean, stddev=self.init_std_dev, seed=self.random_state),
            embeddings_regularizer=tf.keras.regularizers.L2(
                self.reg_impl_embed)
        )
        self.user_bias = keras.layers.Embedding(
            input_dim=self.n_users, output_dim=1,
            embeddings_initializer=tf.keras.initializers.Zeros(),
            embeddings_regularizer=tf.keras.regularizers.L2(self.reg_user_bias)
        )
        self.item_bias = keras.layers.Embedding(
            input_dim=self.n_items, output_dim=1,
            embeddings_initializer=tf.keras.initializers.Zeros(),
            embeddings_regularizer=tf.keras.regularizers.L2(self.reg_item_bias)
        )

    def preprocess(self, X):
        self.user_rated_items = [[] for _ in range(self.n_users)]
        for u, i in zip(X[:, 0], X[:, 1]):
            self.user_rated_items[u].append(i)
        self.user_rated_items = tf.ragged.constant(
            self.user_rated_items, dtype=tf.int32)

    def call(self, inputs):
        user, item = inputs[:, 0], inputs[:, 1]
        user_embed, item_embed = self.user_embedding(
            user), self.item_embedding(item)
        user_bias, item_bias = self.user_bias(user), self.item_bias(item)
        rated_items = tf.gather(self.user_rated_items, user)
        item_lengths = tf.cast(
            tf.map_fn(tf.shape, rated_items).to_tensor(), dtype=tf.float32)

        implicit_embed = self.item_implicit_embedding(rated_items)
        implicit_embed_sum = tf.reduce_sum(implicit_embed, axis=1)
        moderated_implicit_embed = tf.math.divide(
            implicit_embed_sum, tf.math.sqrt(item_lengths))

        total_user_embed = tf.math.add(user_embed, moderated_implicit_embed)
        rating = tf.math.reduce_sum(tf.multiply(
            total_user_embed, item_embed), 1, keepdims=True)
        total_bias = tf.math.add(
            self.global_mean, tf.math.add(user_bias, item_bias))
        rating = tf.math.add(rating, total_bias)

        return rating
