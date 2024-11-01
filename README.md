# Spaceship Titanic: Predicting Passenger Transport

This repository contains a machine learning project to predict whether passengers on the fictional Spaceship Titanic were transported to another dimension. The project was conducted using Python, and various models were tested to optimize prediction accuracy.

## Project Overview

The main objective of this project is to use a combination of **Exploratory Data Analysis (EDA)**, **Feature Engineering**, **Model Training**, and **Hyperparameter Tuning** to build and optimize a predictive model. We experimented with Logistic Regression and XGBoost, applied Grid Search for parameter optimization, and conducted feature selection to enhance model performance.

## Files in This Repository

- **train.csv**: The training dataset provided by Kaggle.
- **test.csv**: The test dataset provided by Kaggle.
- **preprocessed_train_data.csv** and **preprocessed_test_data.csv**: Processed versions of the datasets after handling missing values and feature engineering.
- **titanic_logistic_regression.ipynb**: Jupyter Notebook containing the code for data preprocessing, model training, and evaluation.
- **README.md**: This file, providing an overview of the project and code.

## Project Workflow

### 1. Data Preprocessing and Exploratory Data Analysis (EDA)
In the EDA and preprocessing phase, the following steps were performed:
- **Data Loading**: Loaded training and test datasets and took an initial look at the data structure.
- **Missing Values Analysis**: Identified and imputed missing values based on observed distributions.
  - *HomePlanet* and *Destination*: Imputed with the most frequent values.
  - *Age*: Imputed with the median age.
  - *CryoSleep* and *VIP*: Imputed based on the most frequent values or logical assumptions related to other spending columns.
- **Feature Engineering**: Split the *Cabin* column into three new features: *Deck*, *Position*, and *Side*, and removed unnecessary columns like *PassengerId* and *Name*.

### 2. Feature Engineering
- **Onboard Spending Feature**: Created a new feature, *Onboard Spending*, by summing individual spending categories (*RoomService*, *FoodCourt*, *ShoppingMall*, *Spa*, *VRDeck*). This feature was tested but ultimately did not improve the model significantly.
- **Feature Selection**: Based on coefficient analysis from logistic regression, features with low importance were removed to improve model interpretability and efficiency.

### 3. Model Training
- **Logistic Regression**:
  - **Baseline Model**: Trained a logistic regression model on the processed data to set a baseline for accuracy.
  - **Grid Search**: Applied Grid Search to optimize hyperparameters for regularization (e.g., `C`, `penalty`).
- **XGBoost**:
  - **Basic Model**: Trained an initial XGBoost model on the same processed data.
  - **Grid Search**: Performed hyperparameter tuning on the XGBoost model using Grid Search to improve performance.

### 4. Model Evaluation
- **Validation Accuracy**: Evaluated model performance on the validation set and recorded accuracy for each approach (Logistic Regression vs. XGBoost, with and without feature selection, with and without *Onboard Spending*).
- **Confusion Matrix**: Plotted confusion matrices for each model to analyze prediction errors.
- **Coefficient Analysis**: Converted logistic regression coefficients to Odds Ratios for interpretability.

### Results

| Model                                    | Validation Accuracy |
|------------------------------------------|----------------------|
| Logistic Regression (Basic)              | 0.782               |
| Logistic Regression (Grid Search)        | 0.783               |
| Logistic Regression (OnboardSpending)    | 0.783               |
| Logistic Regression (Feature Selection)  | 0.777               |
| XGBoost (Basic)                          | 0.787               |
| XGBoost (Grid Search)                    | 0.789               |
| XGBoost (OnboardSpending)                | 0.790               |
| XGBoost (Feature Selection)              | 0.776               |

### Summary

- The XGBoost model with the *Onboard Spending* feature and Grid Search achieved the highest validation accuracy of 79.0%.
- Feature selection resulted in a slight decrease in accuracy, suggesting that the selected features provided value for the models.
- Adding *Onboard Spending* slightly improved accuracy for XGBoost but did not affect logistic regression performance significantly.

## Acknowledgements

This project is based on the Spaceship Titanic dataset provided by Kaggle. Thanks to the Kaggle community for providing a collaborative platform for learning and experimentation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
