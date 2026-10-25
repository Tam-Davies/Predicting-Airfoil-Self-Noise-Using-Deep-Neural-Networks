import matplotlib.pyplot as plt


def plot_feature_distributions(df, columns):
    """
    Plot histograms for selected numerical features.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing the features.
    columns : list
        Names of columns to plot.
    """

    for column in columns:
        plt.figure(figsize=(8, 5))

        plt.hist(df[column], bins=30)

        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()


def plot_target_distribution(df, target_column):
    """
    Plot the distribution of the target variable.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing the target.
    target_column : str
        Name of the target column.
    """

    plt.figure(figsize=(8, 5))

    plt.hist(df[target_column], bins=30)

    plt.title("Distribution of Scaled Sound Pressure")
    plt.xlabel("Scaled Sound Pressure (dB)")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()