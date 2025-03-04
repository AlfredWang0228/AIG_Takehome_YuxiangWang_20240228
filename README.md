# AIG Takehome Submission

This project is the submission for the AIG take-home assignment and is intended solely for this purpose.

## Dependencies
The project requires the following dependencies:
- [Ollama](https://ollama.com/)
- [PySpark](https://spark.apache.org/docs/latest/api/python/getting_started/install.html)

Please refer to the official documentation for installation instructions.

## Execution Environment
The project utilizes LLM and embedding model calls, all executed locally. The execution environment is:
- macOS 15.3.1 (24D70)
- M3 Max chip
- 64GB Unified Memory
- Python 3.12

## Project Structure
The project includes the following files and directories:
```
.
├── task1.ipynb
├── task2.ipynb
├── requirements.txt
├── readme.md
└── utils
    ├── __init__.py
    ├── convert_dataset_to_spark.py
    ├── plot_word_counts.py
    ├── section_processing.py
```

During notebook execution, three directories will be generated:
- `parquet_batches_task1`: Stores dataset parquet files for Task 1.
- `parquet_batches_task2`: Stores dataset parquet files for Task 2.
- `screenshots`: Contains necessary visualizations in JPEG format (Task 1).

---

## **Coding Assignment**

### **Dataset:**
IEDGAR SEC filings (public data) - [EDGAR Corpus](https://huggingface.co/datasets/eloukas/edgar-corpus)

### **Language & Implementation:**
- Python
- PySpark

### **Submission Format:**
GitHub repository containing code + plots (JPEG).

### **Expected Maximum Duration:**
3 hours

---

## **Task #1 - Engineering**

### **Objective:**
Given a set of documents, create a solution that allows the end user to visualize the documents in a two-dimensional space and identify outliers.

### **Dataset Specifications:**
- **Year:** 2020
- **Filing Type:** 10-K
- **Sections:** All
- **Companies:** Limit to 10

### **Steps:**
1. Convert the documents to chunks.
2. Convert the chunks into embeddings.
3. Standard scale the embeddings.
4. Perform Principal Component Analysis (PCA).
5. Apply dimensionality reduction.
6. Perform KMeans clustering and assign chunks a cluster number.
7. Create an outlier flag.
8. Generate the following plots:
   - Embeddings in 2D space
   - Colored by assigned clusters
   - Colored by outlier flag
   - Colored by section number

---

## **Task #2 – Gen AI**

### **Dataset Specifications:**
- **Year:** 2018-2020
- **Filing Type:** 10-K
- **Sections:** All
- **Company:** Choose 1
- **Attributes:** Choose 5 data attributes to extract from a single year.

### **Steps:**
1. Convert documents to chunks.
2. Convert chunks to embeddings.
3. Create a query.
4. Create a prompt to extract data from chunks for a specific year.
5. Create a validation dataset (5 true values from chunks).
6. Demonstrate that the LLM can retrieve the correct chunks from the embedding object for the correct year.

---

## **Execution Instructions**
To run the project:
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure Ollama and PySpark are properly set up.
3. Run the Jupyter notebooks (`task1.ipynb` and `task2.ipynb`) step-by-step.

