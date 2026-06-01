# On-The-Fly Processing with Mixture of Factor Analyzers (OTFP-MFA)

This repository contains the core implementation and experimental notebooks for an On-The-Fly Processing (OTFP) system utilizing a streaming Mixture of Factor Analyzers (MFA) algorithm. The system is designed for streaming, clustering, and performing dimensionality reduction on hyperspectral imaging (HSI) data. 

## System Architecture

## Repository Structure

### Core Implementations
* **`mfa.py`**: The base PyTorch implementation of the Mixture of Factor Analyzers (MFA) model. It handles the Expectation-Maximization (EM) steps, latent statistics, and parameter initializations for static model fitting.
* **`otfp.py`**: The streaming wrapper (`MFA_OTFP`) around the core MFA model. It is designed for continuous data streams and handles real-time outlier detection, dynamic component spawning (via DBSCAN), and continuous model updates.

### Test & Benchmarking Notebooks
All results generated for the thesis are based on the notebooks and scripts in the testing directory:
* **`benchmarking.ipynb`**: Compares MFA vs. PCA on reconstruction/compression trade-offs, spectral library matching, and latent factor interpretability.
* **`cloud_test.ipynb`**: Analyzes the model's performance when streaming through cloudy pixels to see if it can map back to underlying materials.
* **`component_spawning.ipynb`**: Rigorously tests the model's ability to dynamically spawn new components when encountering entirely new classes of data (e.g., transitioning from ocean to forest).
* **`environmental_signal_drift.ipynb`**: Tests the system's robustness against gradual signal drift (changes in the mean signal) over time.
* **`noise_drift.ipynb`**: Tests the drift mechanism's response to increasing/changing noise levels across target bands.
* **`interpretability_analysis.ipynb`**: Investigates whether the extracted parameters map accurately to physical phenomena (i.e., true spectral signatures of materials and sensor noise).
* **`memory_usage_benchmarking.py`**: A hardware-viability script used to benchmark memory usage and processing latency over continuous data streams.
* **`train_reference_models.py`**: Utility script for training the baseline PCA and static MFA models used for comparison in the notebooks.

## Data Acquisition & Preprocessing

The hyperspectral data used in this project originates from the HYPSO-1 satellite. 

1. **Download Data**: Navigate to the [HYPSO Data Portal](https://hypso.space/dataportal/en/) and download the required `.nc` (NetCDF) L1A data cubes.
2. **Directory Structure**: Place the downloaded images into the `data/new_data/` directory (you will need to create this folder if it does not exist).
3. **Conversion**: Run the provided conversion script to process the L1A data into L1B format before using it with the models:
   ```bash
   python convert_to_l1b.py

## Usage & Running Tests

### Prerequisites
Make sure you have the required dependencies installed (e.g., `torch`, `scipy`, `scikit-learn`, `matplotlib`, `netCDF4`, `pandas`, `jupyter`).

### Running the Notebooks
To reproduce the thesis results or run the benchmarking scripts:

1. Ensure your converted L1B data is available in the expected data directory.
2. **Important Configuration:** Before running any of the test scripts or notebooks, you must verify that the file paths match the structure of your local machine. Adjust the paths pointing to the `.nc` files accordingly to avoid `FileNotFoundError` issues.
3. Launch Jupyter:
   ```bash
   jupyter notebook
4. Run the notebooks cell-by-cell to view the generated plots and metrics.
