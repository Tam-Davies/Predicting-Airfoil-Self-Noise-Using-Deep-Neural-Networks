from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


def build_baseline_model(input_dim):
    """
    Build the baseline feed-forward neural network.

    Parameters
    ----------
    input_dim : int
        Number of input features.

    Returns
    -------
    tensorflow.keras.Model
        Compiled neural network.
    """

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="linear")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    return model


def build_larger_model(input_dim):
    """
    Build a larger feed-forward neural network.

    Parameters
    ----------
    input_dim : int
        Number of input features.

    Returns
    -------
    tensorflow.keras.Model
        Compiled neural network.
    """

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="linear")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    return model


def build_tuned_model(input_dim):
    """
    Build a tuned feed-forward neural network using
    the larger architecture with a lower learning rate.

    Parameters
    ----------
    input_dim : int
        Number of input features.

    Returns
    -------
    tensorflow.keras.Model
        Compiled neural network.
    """

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="linear")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="mse",
        metrics=["mae"]
    )

    return model