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


class DataPreprocessor:
    """
    Handle feature-target separation, train-test splitting,
    and feature scaling for the airfoil self-noise dataset.
    """

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()

    def split_features_target(self, df):
        """
        Separate input features from the target variable.
        """

        X = df[FEATURE_COLUMNS]
        y = df[TARGET_COLUMN]

        return X, y

    def split_data(self, X, y):
        """
        Split the dataset into training and testing sets.
        """

        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state
        )

    def scale_features(self, X_train, X_test):
        """
        Fit the scaler on training data and transform
        both training and testing features.
        """

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled