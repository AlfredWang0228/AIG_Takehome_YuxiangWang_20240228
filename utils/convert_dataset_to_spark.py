import os
import pandas as pd
from tqdm.auto import tqdm

def convert_dataset_to_spark_df(dataset, spark, batch_size=1000, output_dir="parquet_batches"):
    """
    Convert a HuggingFace dataset to a Spark DataFrame by writing data in batches to Parquet files
    and then loading them with Spark's native Parquet reader.
    
    Parameters:
        dataset (iterable): The HuggingFace dataset to be processed.
        spark (SparkSession): The active SparkSession.
        batch_size (int): The number of records per batch (default is 1000).
        output_dir (str): The directory to store temporary Parquet files (default is "parquet_batches").
    
    Returns:
        pyspark.sql.DataFrame: The combined Spark DataFrame loaded from the Parquet files.
    """
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    batch = []
    file_index = 0

    # Determine the total number of records for the progress bar if possible
    try:
        total = len(dataset)
    except TypeError:
        total = None

    # Write dataset records in batches to Parquet files with progress bar
    for i, record in tqdm(enumerate(dataset), total=total, desc="Writing batches to Parquet"):
        batch.append(record)
        if (i + 1) % batch_size == 0:
            pandas_df = pd.DataFrame(batch)
            output_file = os.path.join(output_dir, f"batch_{file_index}.parquet")
            pandas_df.to_parquet(output_file, index=False)
            file_index += 1
            batch = []

    # Process any remaining records that did not fill a complete batch
    if batch:
        pandas_df = pd.DataFrame(batch)
        output_file = os.path.join(output_dir, f"batch_{file_index}.parquet")
        pandas_df.to_parquet(output_file, index=False)

    # Load all parquet files into a Spark DataFrame
    spark_df = spark.read.parquet(os.path.join(output_dir, "*.parquet"))
    return spark_df
