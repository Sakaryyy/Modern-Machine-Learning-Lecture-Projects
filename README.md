# Modern Machine Learning Lecture Projects

This repository has the course work for the *Modern Machine Learning* lecture.  
The first project focuses on
the [UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) and implements two learning
tasks:

* **Regression** – predict the hourly rental count (`cnt`) with linear/ridge regression.  
  The pipeline performs greedy forward ablation over interpretable feature groups, compares against blind baselines (
  mean count and hour-of-day lookup), and produces detailed excel files and plots.
* **Classification** – predict the hour of day with multinomial logistic regression.  
  Features are normalised/standardised automatically, the model searches over an L2 grid, and the mutual information
  between bike count and the chosen feature set is reported.

## Repository layout

```
Project_1_Bikeshare/
├── config/                # Editable experiment configuration (YAML)
├── data/                  # Raw/processed data caches (created on demand)
├── outputs/               # Figures and tables written by the experiments
└── src/                   # Python package implementing the project
```

Run all commands from the repository root unless stated otherwise.

## Installation

1. **Create and activate a Python environment** (Python 3.10+ recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install the project dependencies**:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   The requirements include `jax`/`jaxlib`. If you want to use a GPU you must ensure that `jax` is installed with the
   correct install flag for your CUDA version (
   see [Jax Installation Guide](https://docs.jax.dev/en/latest/installation.html)).

## Configuration

Project 1 settings live in [`Project_1_Bikeshare/config/experiment.yaml`](Project_1_Bikeshare/config/experiment.yaml).  
Key sections:

* **paths** – output directories relative to the repository. Change these when running experiments on different storage
  locations.
* **splits** – chronological boundaries separating train/validation/test partitions.
* **features** – switches controlling which feature groups enter the regression and classification pipelines (like hour
  encodings, polynomial temperature terms, weather indicators). Can be expanded if new features are added.

The loader automatically resolves relative paths with respect to the repository root. You can also point to an external
config by passing the `EXPERIMENT_CONFIG` environment variable (see below).

## Usage

From the repository root run the project entry point:

```bash
python -m Project_1_Bikeshare.src.main --mode <eda|regression|classify>
```

Optional flags:

* `--force-fetch` - redownload the raw CSV from UCI, bypassing the local cache.
* `--lam-grid` - space-separated ridge penalties for regression ablation.
* `--epsilon` - tolerance used to pick the minimal feature subset within reach of the best validation score.
* `--classify-reg-grid`, `--classify-learning-rate`, `--classify-max-iter`, `--classify-tol` - equivalents for the
  logistic regression experiment.
* `--y-transform` - optionally transform the regression target using `log1p` or `sqrt`.

The workflow for each mode is as follows:

### 1. Exploratory data analysis (`--mode eda`)

* Downloads/caches the dataset (`data/raw`).
* Cleans types, removes leakage columns, and saves a processed copy (`data/processed/bike_clean`).
* Produces time series, histograms, scatter plots, categorical comparisons, and correlation diagnostics under
  `outputs/figures/eda` together with tabular summaries (`outputs/tables/eda`).

### 2. Regression (`--mode regression`)

* Uses the processed dataset and respects the chronological split from the configuration.
* Builds candidate feature groups (hour encodings, temperature polynomials, optional calendar/weather features) and
  performs greedy forward ablation.
* Selects the smallest subset within `(1 + epsilon)` of the best validation RMSE across the lambda grid.
* Fits ridge regression on train+validation, evaluates on the held-out test set, and compares against mean/hour-of-day
  baselines.
* Writes metrics, ablation traces, coefficient tables, predictions, and comparison plots to
  `outputs/tables/model/regression` and `outputs/figures/model/regression`.

### 3. Classification (`--mode classify`)

* Predicts the hour-of-day label via multinomial logistic regression.
* Normalises the bike-count feature into `[0, 1]` and optionally standardises other columns.
* Executes forward ablation to find a minimal feature set within the tolerance, sweeping an L2 grid.
* Reports accuracy, misclassification rate, log-loss, and mutual information, comparing the model against a uniform
  blind guesser.
* Saves confusion matrices, per-hour accuracy, probability calibration, optimisation history, and all tabular outputs
  under `outputs/tables/model/classification` and `outputs/figures/model/classification`.

## Custom configuration file

To run with a custom configuration file without modifying the default YAML, set the `EXPERIMENT_CONFIG` environment
variable to the desired path:

```bash
EXPERIMENT_CONFIG=/path/to/my_config.yaml python -m Project_1_Bikeshare.src.main --mode regression
```
