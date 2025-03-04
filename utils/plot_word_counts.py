"""
plot_word_counts.py

Provides a helper function to:
  1) Identify columns that start with "section_" (but not ending in "_word_count"),
  2) Compute word counts for each section column,
  3) Convert the word counts to a Pandas DataFrame,
  4) Plot histograms in a grid of subplots.
"""

import math
import matplotlib.pyplot as plt

from pyspark.sql.functions import col, split, size
from pyspark.sql import DataFrame

def plot_section_word_counts(
    spark_df: DataFrame,
    n_cols: int = 4,
    bins: int = 20,
    figsize_scale: float = 3.0
):
    """
    Plot histograms of word counts for columns starting with 'section_'.

    :param spark_df: A PySpark DataFrame containing text columns (e.g., 'section_1', 'section_2', etc.).
    :param n_cols: Number of subplots per row.
    :param bins: Number of bins in the histogram.
    :param figsize_scale: scale factor for the figure size.
    """

    # 1) Dynamically get all columns that start with "section_" but not ending in "_word_count"
    section_columns = [
        c for c in spark_df.columns
        if c.startswith("section_") and not c.endswith("_word_count")
    ]

    # 2) For each section column, add a new column that contains the word count
    for s in section_columns:
        spark_df = spark_df.withColumn(s + "_word_count", size(split(col(s), r"\s+")))

    # 3) Create a list of the new word count column names
    word_count_columns = [s + "_word_count" for s in section_columns]

    # 4) Convert the selected word count columns to a Pandas DataFrame
    #    (Ensure your dataset is small enough to convert to Pandas)
    word_counts_pdf = spark_df.select(word_count_columns).toPandas()

    # 5) Determine how many rows of subplots are needed
    n_rows = (len(word_count_columns) + n_cols - 1) // n_cols

    # 6) Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * figsize_scale))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

    # 7) Plot each word count column as a histogram
    for i, wc in enumerate(word_count_columns):
        ax = axes[i]
        word_counts_pdf[wc].hist(ax=ax, bins=bins)
        ax.set_title(wc)
        ax.set_xlabel("Word Count")
        ax.set_ylabel("Frequency")

    # 8) Hide any unused subplots (if the total number of columns isn't a multiple of n_cols)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    # Optional: return the figure/axes if the caller wants further customization
    #return fig, axes
