# Modern Machine Learning Lecture Projects

This repository contains two end-to-end course projects for the *Modern Machine Learning* lecture:

1. **Project 1 – Bikeshare demand modelling.** Uses the
   [UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) to build interpretable
   regression and classification pipelines. Greedy forward ablation over feature groups, ridge/logistic regression
   baselines, and rich reporting artifacts are included by default.
2. **Project 2 – CIFAR-10 image classification.** Implements data analysis, configurable training, inference, and
   automated ablation/hyper-parameter search routines for deep convolutional networks trained on CIFAR-10.

## Repository layout

```
Project_1_Bikeshare/
├── config/                # Editable experiment configuration (YAML)
├── data/                  # Raw/processed data caches (created on demand)
├── outputs/               # Figures and tables written by the experiments
└── src/                   # Python package implementing the project


Project_2_Image_Classification/
└── src/
    ├── config/            # Persistent defaults (paths, models, optimiser settings)
    ├── data_loading/      # CIFAR-10 download, preprocessing and caching utilities
    ├── data_analysis/     # Descriptive statistics and visualisations
    ├── models/            # Baseline MLP and CNN architectures
    ├── training_routines/ # Training loop, optimisers, schedulers, augmentation
    ├── classification_routines/ # Checkpointed inference & reporting
    └── hyperparameter_ablation_experiments/ # Ablation and grid-search orchestration

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

## Project 1 – Bikeshare demand modelling

### Configuration

Project 1 settings live in [`Project_1_Bikeshare/config/experiment.yaml`](Project_1_Bikeshare/config/experiment.yaml).
Key sections:

* **paths** – output directories relative to the repository. Change these when running experiments on different storage
  locations.
* **splits** – chronological boundaries separating train/validation/test partitions.
* **features** – switches controlling which feature groups enter the regression and classification pipelines (like hour
  encodings, polynomial temperature terms, weather indicators). Can be expanded if new features are added.

The loader automatically resolves relative paths with respect to the repository root. You can also point to an external
config by passing the `EXPERIMENT_CONFIG` environment variable (see below).

### Usage

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

#### 1. Exploratory data analysis (`--mode eda`)

* Downloads/caches the dataset (`data/raw`).
* Cleans types, removes leakage columns, and saves a processed copy (`data/processed/bike_clean`).
* Produces time series, histograms, scatter plots, categorical comparisons, and correlation diagnostics under
  `outputs/figures/eda` together with tabular summaries (`outputs/tables/eda`).

#### 2. Regression (`--mode regression`)

* Uses the processed dataset and respects the chronological split from the configuration.
* Builds candidate feature groups (hour encodings, temperature polynomials, optional calendar/weather features) and
  performs greedy forward ablation.
* Selects the smallest subset within `(1 + epsilon)` of the best validation RMSE across the lambda grid.
* Fits ridge regression on train+validation, evaluates on the held-out test set, and compares against mean/hour-of-day
  baselines.
* Writes metrics, ablation traces, coefficient tables, predictions, and comparison plots to
  `outputs/tables/model/regression` and `outputs/figures/model/regression`.

#### 3. Classification (`--mode classify`)

* Predicts the hour-of-day label via multinomial logistic regression.
* Normalises the bike-count feature into `[0, 1]` and optionally standardises other columns.
* Executes forward ablation to find a minimal feature set within the tolerance, sweeping an L2 grid.
* Reports accuracy, misclassification rate, log-loss, and mutual information, comparing the model against a uniform
  blind guesser.
* Saves confusion matrices, per-hour accuracy, probability calibration, optimisation history, and all tabular outputs
  under `outputs/tables/model/classification` and `outputs/figures/model/classification`.

### Custom configuration file

To run with a custom configuration file without modifying the default YAML, set the `EXPERIMENT_CONFIG` environment
variable to the desired path:

```bash
EXPERIMENT_CONFIG=/path/to/my_config.yaml python -m Project_1_Bikeshare.src.main --mode regression
```

## Project 2 – CIFAR-10 image classification

### Configuration

Project-wide defaults are stored in `outputs/configs/project_config.yaml` and are automatically created the first time
you invoke the CLI. The persisted [`ProjectConfig`](Project_2_Image_Classification/src/config/config.py) keeps track of
paths, training hyper-parameters, model defaults, and experiment grids. Key sections include:

* **paths** – data storage (`data/`), training artefacts (`outputs/training_runs/`), config snapshots, model
  checkpoints,
  and analysis/figure directories.
* **training** – baseline hyper-parameters such as optimiser (`adamw`), scheduler (`cosine_decay`), learning rate,
  weight decay, default epoch/batch settings, and whether to enable data augmentation.
* **baseline_defaults** / **cnn_defaults** – architecture definitions for the MLP baseline and CNN classifier (number of
  blocks, hidden units, dropout, activation functions, etc.).
* **ablation** / **hyperparameter_search** – parameter grids explored by the automated ablation study and the
  hyper-parameter search utility.

You can supply a YAML file to any command via `--config`. The file can override the `trainer` section (hyper-parameters
such as optimiser, scheduler, augmentation toggles) and the `model` section (architectural settings). CLI flags always
take precedence over YAML values.

### Usage

All commands share the same entry point and base arguments:

```bash
python -m Project_2_Image_Classification.src.main \
  [--data-dir <path>] [--val-split <float>] [--random-seed <int>] \
  [--log-level <LEVEL>] [--log-dir <path>] [--log-file <name>] \
  <command> [command-specific options]
```

The CLI downloads and caches CIFAR-10 automatically. Train/validation splits are reproducible via `--random-seed`, and
`--val-split` must be strictly between 0 and 1. Artefacts (checkpoints, figures, tables) are stored under
`outputs/` by default.

#### `training`

Train one of the registered models (`baseline`, `cnn`). Command-specific flags include:

* `--model` – architecture identifier.
* `--config` – optional YAML override file (see above).
* `--output-dir` – directory for artefacts, defaults to `outputs/training_runs/<model>_<timestamp>/`.
* `--epochs`, `--batch-size`, `--eval-batch-size`, `--learning-rate`, `--optimizer`, `--scheduler` – override defaults
  without editing YAML files.
* `--enable-data-augmentation` / `--disable-data-augmentation` – toggle the augmentation pipeline globally.
* `--enable-{random-crop|horizontal-flip|vertical-flip|rotation|color-jitter|gaussian-noise|cutout}` and corresponding
  `--disable-*` flags – switches for individual augmentation steps.

Each training run saves checkpoints (`checkpoints/final_params.msgpack`), a model definition (`model_definition.yaml`),
training curves/figures, Excel & CSV histories, and a JSON metrics summary inside the chosen output directory.

#### `classification`

Evaluate a saved checkpoint on the test split (or the dataset provided via `--data-dir`). Required and optional flags:

* `--checkpoint` – path to either the run directory or the `.msgpack` file. The CLI resolves the enclosing run folder
  automatically.
* `--input-path` – optional directory with extra images (currently emits a warning and defaults to the test split).
* `--batch-size` – evaluation batch size (falls back to the original training setting when omitted).
* `--no-predictions` – skip writing per-sample predictions.
* `--output-dir` – override the default `classification/` subdirectory next to the run.

The routine recreates the model from `model_definition.yaml`, reports metrics, saves confusion matrices, prediction
galleries, per-class accuracies, confidence histograms, optional feature visualisations, and (unless disabled) a CSV of
predictions.

#### `analysis`

Produce a descriptive report for CIFAR-10:

* `--output-dir` – where to store figures and JSON summaries (defaults to `<data-dir>/analysis/`).
* `--sample-seed` – controls the random gallery of training samples.

The analyzer summarises split sizes, label balance, pixel statistics, per-channel moments, and saves illustrative
figures (split overview charts, sample grids, class distributions).

#### `experiments`

Automate ablations and hyper-parameter sweeps over the CNN architecture:

* `--mode` – choose `ablation`, `hyperparameter`, or `both` (default) to run the respective routines.
* `--model` – starting architecture (defaults to `baseline`).
* `--config` – YAML overrides applied to both the trainer and the experimental model template.
* `--output-dir` – base directory for artefacts (defaults to the training run location defined in the config).
* `--enable-data-augmentation` / `--disable-data-augmentation` plus the per-augmentation toggles listed under
  `training` – propagate augmentation policies to all experimental runs.

Results are organised beneath `outputs/analysis/` by default, with subdirectories for ablation traces and
hyper-parameter
searches. Each run reuses the training infrastructure, saving metrics, histories, and model definitions for comparison.
