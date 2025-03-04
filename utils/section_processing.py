"""
section_processing.py

Provides a SectionProcessor class to handle:
- Splitting text into sentences with Spark NLP,
- Chunking sentences via a PySpark UDF,
- Expanding chunk arrays into multiple columns,
- Removing intermediate columns.
"""

import pyspark
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, size, expr, udf
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml import Pipeline
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import SentenceDetector
from tqdm.auto import tqdm


class SectionProcessor:
    """
    - chunking logic for text sections,
    - Spark NLP pipeline for sentence detection,
    - DataFrame transformations to expand chunk arrays,
    - optional parameters for chunk size, overlap, etc.
    """

    def __init__(self, max_words=500, overlap=1):
        """
        Initialize the SectionProcessor with desired parameters.
        
        :param max_words: Maximum number of words allowed in a chunk
        :param overlap: Number of overlapping sentences retained
                        between consecutive chunks.
        """
        self.max_words = max_words
        self.overlap = overlap
        
        # Create a UDF once for chunking. This references the pure Python method below.
        self.chunk_sentences_udf = udf(
            self._chunk_sentences_func, 
            ArrayType(StringType())
        )

    def _chunk_sentences_func(self, sentences):
        """
        Pure Python function that takes an array of sentences and chunks them
        into multiple strings, each containing up to `max_words` words.
        Retains `overlap` sentences at the transition between chunks.

        :param sentences: List of sentences
        :return: A list of chunked sentences (each chunk is a large string).
        """
        if not sentences:
            return []

        chunks = []
        current_chunk = []
        current_count = 0

        for sentence in sentences:
            word_count = len(sentence.split())

            # If adding this sentence to current_chunk exceeds max_words
            if current_chunk and (current_count + word_count > self.max_words):
                # Append the current chunk as a single string
                chunks.append(" ".join(current_chunk))

                # Retain overlap number of sentences at the chunk boundary
                if self.overlap > 0 and len(current_chunk) >= self.overlap:
                    current_chunk = current_chunk[-self.overlap:]
                else:
                    current_chunk = []
                current_count = sum(len(s.split()) for s in current_chunk)

            # Add the new sentence
            current_chunk.append(sentence)
            current_count += word_count

        # Append any leftover chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def process_section_column(self, df: DataFrame, section: str) -> DataFrame:
        """
        Perform the following steps on a single 'section' column:
          1. Convert text in 'section' to Spark NLP documents,
          2. Split documents into an array of sentence annotations,
          3. Extract actual sentence texts (strings),
          4. Use UDF to chunk these sentences,
          5. Record the number of chunks per row.

        :param df: A PySpark DataFrame
        :param section: The name of the column containing text for a given section
        :return: A PySpark DataFrame with new columns for doc, sentences, chunk array, chunk count
        """

        assembler = DocumentAssembler() \
            .setInputCol(section) \
            .setOutputCol(f"{section}_doc")

        sent_detector = SentenceDetector() \
            .setInputCols([f"{section}_doc"]) \
            .setOutputCol(f"{section}_sentences")

        pipeline = Pipeline(stages=[assembler, sent_detector])
        df = pipeline.fit(df).transform(df)

        # Extract the actual sentence text from the annotations
        df = df.withColumn(
            f"{section}_sentences_text",
            expr(f"transform({section}_sentences, x -> x.result)")
        )

        # Apply UDF to chunk the sentences
        df = df.withColumn(
            f"{section}_chunks",
            self.chunk_sentences_udf(col(f"{section}_sentences_text"))
        )

        # Count how many chunks were created
        df = df.withColumn(
            f"{section}_chunk_count",
            size(col(f"{section}_chunks"))
        )

        return df

    def expand_chunks(self, df: DataFrame, section: str) -> DataFrame:
        """
        Expand the array of chunks (in column '{section}_chunks') into separate columns.
        For example, {section}_chunk_1, {section}_chunk_2, etc.

        :param df: A PySpark DataFrame
        :param section: The name of the section column (e.g., 'section_1').
        :return: A PySpark DataFrame with expanded chunk columns.
        """
        row_with_max = df.agg({f"{section}_chunk_count": "max"}).collect()[0][0]
        # If no data or 0 chunk_count, skip expansion
        if row_with_max is None or row_with_max == 0:
            return df
        
        max_chunks = int(row_with_max)
        for i in range(max_chunks):
            df = df.withColumn(
                f"{section}_chunk_{i+1}",
                col(f"{section}_chunks")[i]
            )

        return df

    def drop_intermediate(self, df: DataFrame, section: str) -> DataFrame:
        """
        Drop intermediate columns from the pipeline: doc, sentences, sentences_text, chunks, chunk_count.

        :param df: A Spark DataFrame
        :param section: The base name of the section column
        :return: A PySpark DataFrame without the intermediate columns.
        """
        cols_to_drop = [
            f"{section}_doc",
            f"{section}_sentences",
            f"{section}_sentences_text",
            f"{section}_chunks",
            f"{section}_chunk_count",
        ]
        return df.drop(*cols_to_drop)

    def process_all_sections(
        self, 
        df: DataFrame, 
        section_cols=None, 
        show_progress: bool = False,
        force_action: bool = False
    ) -> DataFrame:
        """
        Top-level method to process multiple 'section_' columns.

        :param df: The Spark DataFrame
        :param section_cols: A list of specific columns to process. 
                             If None, automatically picks columns starting with 'section_'.
        :param show_progress: If True, uses tqdm to show a progress bar over column iteration.
        :param force_action: If True, calls df.count() after processing each column 
                             (forces Spark to run that step and updates the tqdm bar).
        :return: The processed DataFrame
        """
        if section_cols is None:
            # default to columns starting with "section_"
            section_cols = [c for c in df.columns if c.startswith("section_")]

        # Prepare the iterable for tqdm
        if show_progress:
            columns_iterable = tqdm(section_cols, desc="Processing sections")
        else:
            columns_iterable = section_cols

        for section in columns_iterable:
            df = self.process_section_column(df, section)
            df = self.expand_chunks(df, section)
            df = self.drop_intermediate(df, section)

            # Optionally force an action so that you see each iteration's progress in real time
            if force_action:
                df.count()

        return df
