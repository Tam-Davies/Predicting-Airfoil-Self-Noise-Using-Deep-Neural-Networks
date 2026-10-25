from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


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
        Dense(32, activation="relu", input_shape=(input_dim,)),
        Dense(16, activation="relu"),
        Dense(1, activation="linear")
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model