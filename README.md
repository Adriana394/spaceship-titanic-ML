# Spaceship Titanic: Predicting Passenger Transport

This repository contains a Kaggle machine learning project for predicting whether passengers on the fictional Spaceship Titanic were transported to another dimension. The project focuses on exploratory data analysis, preprocessing, feature engineering, model comparison, and hyperparameter tuning.

## Project Overview

The workflow compares Logistic Regression and XGBoost models for a binary classification task. It is organized as three notebooks — exploration, preprocessing, and modelling — and covers missing-value handling, cabin feature extraction, spending-based feature engineering, feature selection, Grid Search, and validation-set evaluation.

## Repository Structure

```text
.
|-- data/
|   |-- train.csv               # raw Kaggle training data
|   |-- test.csv                # raw Kaggle test data
|   |-- processed_train.csv     # cleaned data (produced by notebook 02)
|   `-- processed_test.csv      # cleaned data (produced by notebook 02)
|-- notebooks/
|   |-- 01_eda.ipynb            # exploratory data analysis
|   |-- 02_preprocessing.ipynb  # cleaning & feature engineering
|   `-- 03_modeling.ipynb       # model training & comparison
|-- requirements.txt
|-- README.md
`-- LICENSE
```

## Notebooks

1. **`01_eda.ipynb` — Exploratory Data Analysis.** Inspects data types, missing values, the target distribution, the `Cabin` column, numeric feature distributions, and the relationship between key features and the target. This notebook is read-only and does not modify the data.
2. **`02_preprocessing.ipynb` — Preprocessing & Feature Engineering.** Drops non-predictive columns, splits `Cabin` into `Deck`/`Number`/`Side`, imputes missing values (all statistics computed on the training set only), engineers the `OnboardSpending` feature, and writes the cleaned data to `data/processed_*.csv`.
3. **`03_modeling.ipynb` — Modelling.** Trains and compares Logistic Regression and XGBoost across eight experiments (baseline, Grid Search, `OnboardSpending`, and feature selection), with confusion-matrix and odds-ratio interpretability for the tuned logistic regression.

### Avoiding data leakage

Encoding and scaling are not applied to the full dataset before splitting. Instead, preprocessing and the estimator are combined into a single scikit-learn `Pipeline` that is fit only on the training split — and refit on each fold inside `GridSearchCV` — so the validation set never influences the fitted transformers.

## Results

Validation accuracy on a held-out 20% split:

| Model                                   | Validation Accuracy |
|-----------------------------------------|---------------------|
| Logistic Regression (Basic)             | 0.795               |
| Logistic Regression (Grid Search)       | 0.795               |
| Logistic Regression (OnboardSpending)   | 0.795               |
| Logistic Regression (Feature Selection) | 0.788               |
| XGBoost (Basic)                         | 0.802               |
| XGBoost (Grid Search)                   | 0.809               |
| XGBoost (OnboardSpending)               | 0.816               |
| XGBoost (Feature Selection)             | 0.803               |

The best validation result was XGBoost with the `OnboardSpending` feature at 81.6% validation accuracy.

## How to Run

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Launch Jupyter and run the notebooks in order:

```bash
jupyter lab
```

Run `01_eda.ipynb`, then `02_preprocessing.ipynb` (which writes the processed CSVs), then `03_modeling.ipynb`.

## Notes and Limitations

- This is an early applied ML project and should be read as a documented learning project, not a production pipeline.
- Imputation statistics are computed on the full training set. For a stricter setup they would be fit per cross-validation fold; encoding and scaling already are.
- The feature engineering decisions are useful for experimentation, but should be validated more rigorously before reuse in a real application.
- Accuracy is reported for learning and comparison. For higher-stakes classification tasks, additional metrics and validation strategies would be needed.

## Acknowledgements

This project is based on the Spaceship Titanic dataset provided by Kaggle.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
