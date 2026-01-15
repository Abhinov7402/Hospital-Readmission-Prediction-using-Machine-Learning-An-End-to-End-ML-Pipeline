# Hospital-Readmission-Prediction-using-Machine-Learning-An-End-to-End-ML-Pipeline
End-to-end machine learning pipeline to predict hospital readmissions using real-world healthcare data. Includes data cleaning, feature engineering, patient-level aggregation, and benchmarking of multiple models (logistic regression, penalized GLM, random forest, neural networks, MARS) with cross-validated evaluation using LogLoss, Accuracy, Kappa

Kaggle competetion link: https://www.kaggle.com/competitions/2024-dsa-ise-ida-classification-hw-7/submissions#

My Team Name : (C) HCOONaH Matata 9

Secured 15th place among 39 competing teams on the leaderboard



The dataset consists of hospital encounter–level records for patients, with the objective of predicting whether a patient will be readmitted. Separate training and test datasets were provided.

## Data Cleaning and Missing Value Handling

Rows with missing patientID values in the test dataset were removed, as patient identifiers are required for aggregation and prediction.

Numeric variables were imputed using the median, ensuring robustness to outliers.

Categorical variables were imputed using a constant level ("Unknown") during initial preprocessing and later refined using mode imputation after aggregation.

Columns with a single unique value across all observations (examide, citoglipton, glimepiride.pioglitazone) were removed, as they provide no predictive information.

## Feature Engineering

The target variable readmitted was converted into a binary factor with levels "yes" and "no".

The original age categories were consolidated into three clinically meaningful groups:

Young: 0–30

Middle-aged: 30–60

Old: 60+

Several numerically coded categorical variables (e.g., admission_type, admission_source, discharge_disposition) were explicitly converted into factor variables with consistent levels across training and test data to prevent unseen-level issues during prediction.

## Patient-Level Aggregation

Since patients may have multiple hospital encounters, the data were aggregated to the patient level:

Numeric features were summarized using the mean across encounters.

Categorical features were summarized using the mode (most frequent value).
This aggregation reduces noise from repeated encounters and aligns the modeling task with patient-level readmission risk.

## Post-Aggregation Imputation

Remaining missing numeric values after aggregation were again imputed using the median.

Missing categorical values were imputed using the mode.

Missing discharge_disposition values were assigned a clinically common category to maintain factor consistency.

## Modeling Approach
Evaluation Strategy

All models were trained using cross-validation to ensure generalizable performance:

Primarily 5-fold cross-validation was used.

Models were evaluated using:

LogLoss (primary metric, emphasizing probability quality),

Accuracy,

Cohen’s Kappa, to account for potential class imbalance.

Custom summary functions were implemented to compute these metrics consistently across models.

Models Implemented
# 1. Logistic Regression (Baseline)

A multivariable logistic regression model was used as a baseline due to its interpretability and suitability for binary classification.

Predictors included demographics, admission characteristics, medication usage, and prior utilization metrics.

Model performance was evaluated using cross-validated LogLoss, Accuracy, and Kappa.

# 2. Penalized Logistic Regression

Regularized logistic regression models were trained using Elastic Net, tuning over a grid of alpha (ridge to lasso) and lambda values.

This approach reduces overfitting and performs implicit feature selection.

LogLoss was used as the optimization metric.

# 3. Decision Tree

A classification tree model was trained to capture nonlinear relationships and interactions.

Cross-validation was used to assess stability and generalization.

# 4. Random Forest

An ensemble of decision trees was trained to improve predictive performance and reduce variance.

The number of variables sampled at each split (mtry) was tuned using cross-validation.

The model was optimized using LogLoss.

# 5. Neural Networks

Feedforward neural networks were trained using the nnet package.

Cross-validation was applied to assess Accuracy and Kappa.

Neural networks provided a nonlinear modeling alternative to tree-based methods.

# 6. k-Nearest Neighbors (kNN)

A distance-based classifier was implemented to benchmark performance against instance-based learning.

Cross-validation was used to select optimal parameters.

# 7. MARS (Multivariate Adaptive Regression Splines)

MARS models were trained to capture nonlinearities and interactions while maintaining interpretability.

Model complexity was controlled via pruning parameters.

Classification was performed using a binomial link function.

# Prediction and Output

For each trained model:

Probabilities of hospital readmission (P(readmitted = yes)) were generated for the test dataset.

Final outputs were saved as CSV files containing patientID and predicted readmission probabilities, suitable for leaderboard submission or downstream decision support.
