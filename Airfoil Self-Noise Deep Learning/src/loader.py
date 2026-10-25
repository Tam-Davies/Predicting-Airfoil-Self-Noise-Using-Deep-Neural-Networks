import pandas as pd


COLUMN_NAMES = [
    "frequency_hz",
    "angle_of_attack_deg",
    "chord_length_m",
    "free_stream_velocity_ms",
    "suction_side_displacement_thickness_m",
    "scaled_sound_pressure_db",
]


def load_airfoil_data(file_path):
    """
    Load the NASA Airfoil Self-Noise dataset.

    Parameters
    ----------
    file_path : str
        Path to the dataset file.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset with descriptive column names.
    """

    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None
    )

    df.columns = COLUMN_NAMES

    return df