# Spaceship Titanic: Predicting Passenger Transport

This repository contains a Kaggle machine learning project for predicting whether passengers on the fictional Spaceship Titanic were transported to another dimension. The project focuses on exploratory data analysis, preprocessing, feature engineering, model comparison, and hyperparameter tuning.

## Project Overview

The workflow compares Logistic Regression and XGBoost models for a binary classification task. It includes missing-value handling, cabin feature extraction, spending-based feature engineering, feature selection, Grid Search, and validation-set evaluation.

## Repository Structure

```text
.
|-- Data/
|   |-- train.csv
|   |-- test.csv
|   |-- preprocessed_train_data
|   `-- preprocessed_test_data
|-- titanic_logistic_regression_xgb.ipynb
|-- requirements.txt
|-- README.md
`-- LICENSE
```

## Workflow

1. Load the Kaggle training and test datasets from `Data/`.
2. Explore missing values, categorical variables, passenger spending, and target distribution.
3. Impute missing values using distribution-based assumptions and median or mode strategies.
4. Split the `Cabin` column into `Deck`, `Position`, and `Side`.
5. Create and evaluate an `OnboardSpending` feature from spending columns.
6. Train Logistic Regression and XGBoost models.
7. Tune hyperparameters with Grid Search.
8. Compare validation accuracy, confusion matrices, and coefficient-based interpretability.

## Results

| Model                                   | Validation Accuracy |
|-----------------------------------------|---------------------|
| Logistic Regression (Basic)             | 0.782               |
| Logistic Regression (Grid Search)       | 0.783               |
| Logistic Regression (OnboardSpending)   | 0.783               |
| Logistic Regression (Feature Selection) | 0.777               |
| XGBoost (Basic)                         | 0.787               |
| XGBoost (Grid Search)                   | 0.789               |
| XGBoost (OnboardSpending)               | 0.790               |
| XGBoost (Feature Selection)             | 0.776               |

The best validation result in this experiment was XGBoost with the `OnboardSpending` feature at 79.0% validation accuracy.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Open and run the notebook:

```bash
jupyter notebook titanic_logistic_regression_xgb.ipynb
```

## Notes and Limitations

- This is an early applied ML project and should be read as a documented learning project, not a production pipeline.
- The feature engineering decisions are useful for experimentation, but should be validated more rigorously before reuse in a real application.
- Accuracy is reported for learning and comparison. For higher-stakes classification tasks, additional metrics and validation strategies would be needed.

## Acknowledgements

This project is based on the Spaceship Titanic dataset provided by Kaggle.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
