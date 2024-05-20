
##Fraud Detection Project
->Project Description

This project aims to detect fraudulent transactions using various machine learning algorithms. The dataset used for this project contains details of financial transactions and labels indicating whether a transaction is fraudulent or not.

->Dataset

The dataset used in this project is Fraud.csv. The dataset contains various features, including transaction amounts, balances before and after the transactions, and the type of transaction.

->Preprocessing Steps:-
*Handling Missing Values:

Missing values are filled using the median for numerical columns and the mode for categorical columns.
Removing Duplicates:

No duplicates were found in the dataset.
Dropping Unnecessary Columns:

The columns isFlaggedFraud, type, nameDest, and nameOrig are removed due to irrelevance or having only a single value.
Outlier Detection and Removal:

Histograms and box plots are used to visualize the data distribution and identify outliers.
Outliers are capped using the Interquartile Range (IQR) method with the Winsorizer from feature_engine.
Multicollinearity Check:

A correlation matrix is used to identify highly correlated pairs of features.
Features with a correlation above a threshold of 0.9 are dropped to reduce multicollinearity.
Feature Importance
A Random Forest model is used to identify the importance of features. The features are then sorted by their importance.

->Machine Learning Models:
##Random Forest Classifier
Data Splitting:

The dataset is split into training and testing sets with an 80-20 split.
Model Training:

A Random Forest Classifier is trained on the imputed training data.
Evaluation:

Accuracy and feature importance are evaluated on the training and testing sets.
##Logistic Regression
Data Splitting:

Similar to the Random Forest model, the dataset is split into training and testing sets.
Model Training:

Logistic Regression is trained on the imputed training data.
Evaluation:

Accuracy and classification reports are generated for the test set.
##AdaBoost Classifier
Data Splitting:

The dataset is split into training and testing sets.
Model Training:

AdaBoost Classifier is trained on the training data.
Evaluation:

Accuracy and classification reports are generated for the test set.
Model Evaluation
##Confusion Matrix:

Confusion matrices are generated to evaluate the performance of each model.
Classification Report:

Classification reports, including precision, recall, and F1-score, are generated.
##ROC-AUC Curve:

ROC curves and AUC scores are plotted to evaluate the models' performance in distinguishing between classes.
->>Results:
The Random Forest and Logistic Regression models were evaluated, with the Random Forest model providing feature importance metrics.
ROC-AUC curves were plotted for the models to visualize their performance.
Logistic Regression and Random Forest models showed competitive performance with high accuracy and AUC scores.
Conclusion
This project successfully implemented and evaluated several machine learning models for fraud detection. Future improvements could involve hyperparameter tuning and trying additional models to improve performance further.

Dependencies
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
feature_engine
Installation
To install the necessary packages, run:

bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn feature_engine
Usage
To run the project, execute the script in a Python environment with the necessary dependencies installed.








