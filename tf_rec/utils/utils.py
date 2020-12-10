import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold


def preprocess_and_split(df, shuffle=True, split=True, test_size=0.2, random_state=None):
    """Preprocesses and Splits the dataset if necessary

    Prepares the dataset to be passed into the model
    - Encodes User and Item, each from 0 to len(user/item)
    - Splits into Train and Test and shuffles if shuffle = True
    - Returns Dataset as numpy arrays and Encoding of users and items

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        dataframe of shape (m,3) with columns in the order - user, item, rating
    shuffle : boolean
        boolean to determine should the dataset be shuffled after encoding (default True)
    split : boolean
        boolean to determine if the dataset should be split into train and test set (default True)
    test_size : float
        size of test split (default 0.2)
    random_state : integer
        seed variable, useful for reproducing results (default None)

    Returns
    -------
    Tuple : dataset, user_item_encodings
            dataset contains tuple of train and test variables if shuffle = True
            dataset - (x_train, y_train), (x_test, y_test)
            user_item_encodings - (user_to_encoded, encoded_to_user,item_to_encoded, encoded_to_item)
    """

    # Get unique ids
    user_ids = df.iloc[:, 0].unique().tolist()
    item_ids = df.iloc[:, 1].unique().tolist()

    # Create encoding and decoding maps for users
    user_to_encoded = {user_id: encoded_id for encoded_id,
                       user_id in enumerate(user_ids)}
    encoded_to_user = {encoded_id: user_id for encoded_id,
                       user_id in enumerate(user_ids)}

    # Create encoding and decoding maps for items
    item_to_encoded = {item_id: encoded_id for encoded_id,
                       item_id in enumerate(item_ids)}
    encoded_to_item = {encoded_id: item_id for encoded_id,
                       item_id in enumerate(item_ids)}

    user_item_encodings = (user_to_encoded, encoded_to_user,
                           item_to_encoded, encoded_to_item)

    # Encode Dataset
    df_encoded = pd.DataFrame([], columns=['user', 'item', 'rating'])
    df_encoded['user'] = df.iloc[:, 0].map(user_to_encoded)
    df_encoded['item'] = df.iloc[:, 1].map(item_to_encoded)
    df_encoded['rating'] = df.iloc[:, 2].values.astype(np.float32)

    # Shuffle if shuffle = True
    if shuffle:
        df_encoded = df_encoded.sample(frac=1, random_state=None)

    # Dataframe -> Numpy Arrays
    x = df_encoded[['user', 'item']].values
    y = df_encoded[['rating']].values
    dataset = x, y

    # Split if split = True
    if split:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )
        dataset = (x_train, y_train), (x_test, y_test)

    return dataset, user_item_encodings


def cross_validation(model, X, y, folds=5, epochs=10, batch_size=32, callbacks=None, random_state=None):
    """Performs K-Fold cross validation on data

    Parameters
    ----------
    model : tf.keras.Model
        a compiled model with appropriate metrics to keep track of
    X : numpy.ndarray
        numpy array of shape (m, 2) of user-item pairs
    y : numpy.ndarray
        numpy array of shape (m,) of target rating
    folds : integer
        number of splits for K Fold (default 5)
    epochs : integer
        epochs to fit the model (default 10)
    batch_size : integer
        batch_size to be passed in while fitting (default 32)
    callbacks : list
        list of callbacks to be passed in while fitting (default None)
    random_state : integer
        seed variable, useful for reproducing results (default None)

    Returns
    -------
    numpy.ndarray
        numpy array of shape (folds, 1(loss) + no_of_metrics)
        contains validation scores evaluated for each fold
    """

    # Initalize KFold
    kfolds = KFold(n_splits=folds, random_state=random_state)
    all_metrics = []

    # To build the model
    if type(model).__name__ == 'SVDpp':
        model.implicit_feedback(X[:10,:])
    model(X[:10,:])

    # Workaround to reset weights after each fold fit
    weights = model.get_weights()
    i = 1

    for train, val in kfolds.split(X, y):

        # Gather implicit feedback if model is SVD++
        if type(model).__name__ == 'SVDpp':
            model.implicit_feedback(X[train])

        # Train and evaluate metrics
        history = model.fit(X[train], y[train], batch_size=batch_size, epochs=epochs, validation_data=(X[val], y[val]),
                            callbacks=callbacks)
        fold_score = model.evaluate(X[val], y[val])
        all_metrics.append(fold_score)

        if folds != i:
          model.set_weights(weights)
        
        i += 1

    all_metrics = np.array(all_metrics)

    for i, metric in enumerate(model.metrics_names):
        print(f'Mean {metric.capitalize()} : {np.mean(all_metrics.T[i])}')

    return all_metrics
