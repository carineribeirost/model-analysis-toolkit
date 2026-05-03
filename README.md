# Model Analysis Toolkit

A modular Python toolkit for chemical model assessment, featuring Applicability Domain (AD) analysis and Ensemble SHAP explanations.

## 🚀 Features

### 1. Applicability Domain (AD) Analysis
- **KNN Mahalanobis Distance**: Calculates the distance to the $k=10$ nearest neighbors in the chemical space.
- **Leverage Analysis**: Computes the hat matrix diagonal to identify samples with high influence on the model.
- **Williams Plot**: High-quality scatter plots (Leverage vs. KNN Distance) with automatic threshold detection.
- **Metric Distribution**: Histograms and summary bar charts for domain assessment.

### 2. SHAP Analysis
- **Ensemble Averaging**: Supports averaging SHAP values across multiple models (e.g., cross-validation folds).
- **Automatic Class Handling**: Detects and extracts the positive class for binary classification tasks.
- **Normalized Beeswarm**: Specialized visualization for unbalanced datasets using standardized SHAP values.
- **Local Explanations**: Generates interactive HTML force plots for individual samples.

---

## 🛠️ Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast environment and dependency management.

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd model-analysis-toolkit
   ```

2. **Sync the environment**:
   ```bash
   uv sync
   ```

---

## 💻 Usage

### 1. Applicability Domain Analysis
Run the AD analysis by providing the training set, evaluation set(s), and the list of descriptor columns.

```bash
uv run python -m src.applicability_domain.main \
    --train path/to/internal_descriptors.csv \
    --eval path/to/external_descriptors.csv \
    --descriptors desc1 desc2 desc3 ... \
    --output-dir ad_results
```

**Key Arguments**:
- `--train`: Path to the CSV used to build the domain.
- `--eval`: One or more paths to evaluation CSVs.
- `--descriptors`: Space-separated list of column names to use as features.
- `--neighbors`: Number of neighbors for KNN (default: 10).

### 2. SHAP Analysis
Run the ensemble SHAP analysis by providing the model pickle files and the evaluation data.

```bash
uv run python -m src.shap_analysis.main \
    --models model_fold0.pkl model_fold1.pkl ... \
    --data data_to_explain.csv \
    --target target_column_name \
    --task classification \
    --output-dir shap_results
```

**Key Arguments**:
- `--models`: Path(s) to `.pkl` model files (supports wildcards like `models/*.pkl`).
- `--data`: Path to the evaluation CSV.
- `--target`: Name of the target column to drop from features.
- `--task`: `classification` or `regression` (default: classification).
- `--force-samples`: Number of interactive local force plots to generate (default: 5).

---

## 🧪 Testing

The toolkit includes a comprehensive test suite using `pytest`.

To run all tests:
```bash
uv run pytest
```

---

## 📊 Outputs

- **CSV Results**: Detailed assessment for every molecule.
- **Plots**:
    - `ad_williams_plot.png`: Leverage vs KNN Distance.
    - `distributions.png`: Metric histograms.
    - `summary_bars.png`: Domain count summary.
    - `shap_importance.png`: Global feature importance.
    - `shap_beeswarm.png`: Standard SHAP impact plot.
    - `shap_beeswarm_normalized.png`: Impact plot for unbalanced data.
    - `force_plots/`: Interactive HTML local explanations.

---

