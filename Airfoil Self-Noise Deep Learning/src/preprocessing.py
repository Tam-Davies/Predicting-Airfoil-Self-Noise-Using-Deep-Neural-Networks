import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "frequency_hz",
    "angle_of_attack_deg",
    "chord_length_m",
    "free_stream_velocity_ms",
    "suction_side_displacement_thickness_m",
]

TARGET_COLUMN = "scaled_sound_pressure_db"


def split_features_target(df):
    """
    Separate the input features from the target variable.

    Parameters
    ----------
    df : pandas.DataFrame
        Airfoil Self-Noise dataset.

    Returns
    -------
    X : pandas.DataFrame
        Input features.
    y : pandas.Series
        Target variable.
    """

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Parameters
    ----------
    X : pandas.DataFrame
        Input features.
    y : pandas.Series
        Target variable.
    test_size : float
        Proportion reserved for testing.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )


def scale_features(X_train, X_test):
    """
    Standardize input features using the training data.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training features.
    X_test : pandas.DataFrame
        Testing features.

    Returns
    -------
    X_train_scaled : numpy.ndarray
        Scaled training features.
    X_test_scaled : numpy.ndarray
        Scaled testing features.
    scaler : StandardScaler
        Fitted scaler.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler