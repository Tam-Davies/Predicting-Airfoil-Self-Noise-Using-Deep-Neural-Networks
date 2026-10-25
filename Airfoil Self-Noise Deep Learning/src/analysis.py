import pandas as pd
import matplotlib.pyplot as plt


def create_model_comparison(results):
    """
    Create a comparison table for all trained models.

    Parameters
    ----------
    results : dict
        Dictionary containing model names and their evaluation metrics.

    Returns
    -------
    pandas.DataFrame
        Model comparison table.
    """

    rows = []

    for model_name, metrics in results.items():
        rows.append({
            "Model": model_name,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"]
        })

    comparison_df = pd.DataFrame(rows)

    return comparison_df


def calculate_baseline_changes(comparison_df):
    """
    Calculate percentage changes relative to the baseline model.

    For MAE and RMSE, negative values indicate improvement.
    For R2, positive values indicate improvement.

    Parameters
    ----------
    comparison_df : pandas.DataFrame
        Model comparison table containing a Baseline row.

    Returns
    -------
    pandas.DataFrame
        Comparison table with percentage changes.
    """

    baseline = comparison_df[
        comparison_df["Model"] == "Baseline"
    ].iloc[0]

    comparison_df = comparison_df.copy()

    comparison_df["MAE_change_pct"] = (
        (comparison_df["MAE"] - baseline["MAE"])
        / baseline["MAE"]
    ) * 100

    comparison_df["RMSE_change_pct"] = (
        (comparison_df["RMSE"] - baseline["RMSE"])
        / baseline["RMSE"]
    ) * 100

    comparison_df["R2_change_pct"] = (
        (comparison_df["R2"] - baseline["R2"])
        / baseline["R2"]
    ) * 100

    return comparison_df


def plot_metric_comparison(comparison_df, metric, ylabel, title):
    """
    Plot a comparison of one evaluation metric across models.

    Parameters
    ----------
    comparison_df : pandas.DataFrame
        Model comparison table.
    metric : str
        Column containing the metric to plot.
    ylabel : str
        Label for the y-axis.
    title : str
        Plot title.
    """

    plt.figure(figsize=(10, 6))

    plt.bar(
        comparison_df["Model"],
        comparison_df[metric]
    )

    plt.ylabel(ylabel)
    plt.title(title)

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.tight_layout()
    plt.show()


def plot_learning_curve(history, title):
    """
    Plot training and validation loss across epochs.

    Parameters
    ----------
    history : tensorflow.keras.callbacks.History
        Training history returned by model.fit().
    title : str
        Plot title.
    """

    plt.figure(figsize=(10, 6))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)

    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(y_true, y_pred, title):
    """
    Plot actual target values against model predictions.

    Parameters
    ----------
    y_true : array-like
        Actual target values.
    y_pred : array-like
        Predicted target values.
    title : str
        Plot title.
    """

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.6
    )

    minimum = min(y_true.min(), y_pred.min())
    maximum = max(y_true.max(), y_pred.max())

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.xlabel("Actual Sound Pressure (dB)")
    plt.ylabel("Predicted Sound Pressure (dB)")
    plt.title(title)

    plt.tight_layout()
    plt.show()


def plot_residuals(y_true, y_pred, title):
    """
    Plot prediction residuals against predicted values.

    Parameters
    ----------
    y_true : array-like
        Actual target values.
    y_pred : array-like
        Predicted target values.
    title : str
        Plot title.
    """

    residuals = y_true - y_pred

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_pred,
        residuals,
        alpha=0.6
    )

    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.xlabel("Predicted Sound Pressure (dB)")
    plt.ylabel("Residual (dB)")
    plt.title(title)

    plt.tight_layout()
    plt.show()