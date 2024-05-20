import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score , StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

import xgboost
from xgboost import XGBClassifier
df=pd.read_csv(r"C:\Users\mahen\Downloads\Fraud.csv.crdownload")
df
ddf=df.isnull().sum()
ddf
def fill_na_values(df):
    for column in df.columns:
        if df[column].dtype in [np.float64, np.int64]:  # Numeric columns
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)
        else:  # String columns
            mode_value = df[column].mode()[0]
            df[column].fillna(mode_value, inplace=True)
    return df

# Fill NaN values in the DataFrame
df = fill_na_values(df)
df.duplicated().sum()
#there is no duplicates in my dataset so i go with my processs

#type column is character type and nameOrig and nameDest columns are mixed with numarical and characters and isFlaggedFraud column has only 0 's so i remove these columns from dataset'
df=df.drop(columns = ["isFlaggedFraud","type","nameDest","nameOrig","isFlaggedFraud"], axis = 1)

# Histogram for a single feature
import matplotlib.pyplot as plt
df['amount'].hist(bins=30)
plt.xlabel('amount')
plt.ylabel('Frequency')
plt.show()

# Histogram for all numerical features
df.hist(bins=30, figsize=(15, 10))
plt.show()


#checking outliers
import matplotlib.pyplot as plt
import seaborn as sns

# Box plot for a single feature
sns.boxplot(x=df['amount'])
plt.show()

# Box plot for all numerical features
df.boxplot(figsize=(12, 8))
plt.show()

# we can observe dataframe has outliers so we can remove those outliers and we are removing outliers by columnwise
# Use IQR to identify and cap outliers
Q1 = df['amount'].quantile(0.25)
Q3 = df['amount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Cap outliers
df['amount'] = np.where(df['amount'] > upper_bound, upper_bound, 
                                    np.where(df['amount'] < lower_bound, lower_bound, 
                                             df['amount']))
from feature_engine.outliers import Winsorizer

w1 = Winsorizer(capping_method = "iqr",
                tail = "both",fold = 1.5,
                variables = ["amount"])
df["amount"] = w1.fit_transform(df[['amount']])
plt.boxplot(df.amount)

w1 = Winsorizer(capping_method = "iqr",
                tail = "both",fold = 1.5,
                variables = ["oldbalanceOrg"])
df["oldbalanceOrg"] = w1.fit_transform(df[['oldbalanceOrg']])
plt.boxplot(df.oldbalanceOrg)

w1 = Winsorizer(capping_method = "iqr",
                tail = "both",fold = 1.5,
                variables = ["newbalanceOrig"])
df["newbalanceOrig"] = w1.fit_transform(df[['newbalanceOrig']])
plt.boxplot(df.newbalanceOrig)

w1 = Winsorizer(capping_method = "iqr",
                tail = "both",fold = 1.5,
                variables = ["oldbalanceDest"])
df["oldbalanceDest"] = w1.fit_transform(df[['oldbalanceDest']])
plt.boxplot(df.oldbalanceDest)


w1 = Winsorizer(capping_method = "iqr",
                tail = "both",fold = 1.5,
                variables = ["newbalanceDest"])
df["newbalanceDest"] = w1.fit_transform(df[['newbalanceDest']])
plt.boxplot(df.newbalanceDest)

import matplotlib.pyplot as plt
import seaborn as sns

# Box plot for a single feature
sns.boxplot(x=df['amount'])
plt.show()

# Box plot for all numerical features
df.boxplot(figsize=(12, 8))
plt.show()


## removing multicolinearity

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Display the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.show()

# Set the correlation threshold
threshold = 0.9

# Find pairs of highly correlated variables
highly_correlated_pairs = [(i, j) for i in correlation_matrix.columns for j in correlation_matrix.columns 
                           if i != j and abs(correlation_matrix.loc[i, j]) > threshold]

print("Highly correlated pairs:")
for pair in highly_correlated_pairs:
    print(pair)
# Get the list of columns to drop
columns_to_drop = set([pair[1] for pair in highly_correlated_pairs])

# Drop the columns
df_reduced = df.drop(columns=columns_to_drop, axis=1)

print("Columns dropped due to high correlation:", columns_to_drop)

## ML model building
y=df_reduced.isFraud     #taking target variable as "isFraud"
x=df_reduced.drop(columns = ['isFraud'], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2 ,  random_state = 5)
x_train.shape,y_train.shape
model_r = RandomForestClassifier()
model_r.fit(x_train, y_train)
train_predict = model_r.predict(x_train)
test_predict = model_r.predict(x_test)
model_r.feature_importances_
dfs = pd.DataFrame(model_r.feature_importances_
                   ,index = df_reduced.columns[1:], columns = ["Importances"]).sort_values(by = "Importances", ascending = False)

# Create an imputer with a strategy (e.g., mean, median, etc.)
imputer = SimpleImputer(strategy='mean')

# Fit and transform the imputer on training data
x_train_imputed = imputer.fit_transform(x_train)
x_test_imputed = imputer.transform(x_test)

# Fit the Random Forest model
model_r.fit(x_train_imputed, y_train)

# Make predictions on the test set
predictions = model_r.predict(x_test_imputed)

# Evaluate accuracy
accuracy = accuracy_score(predictions, y_test)
print(f"Accuracy: {accuracy}")


model_lr = LogisticRegression()
model_lr.fit(x_train, y_train)
train_predict = model_lr.predict(x_train)


# Create an imputer with a strategy (e.g., mean, median, etc.)
imputer = SimpleImputer(strategy='mean')

# Fit and transform the imputer on training data
x_train_imputed = imputer.fit_transform(x_train)
x_test_imputed = imputer.transform(x_test)

# Fit the Logistic Regression model
model_lr.fit(x_train_imputed, y_train)

# Make predictions on the test set
predictions = model_lr.predict(x_test_imputed)

# Evaluate accuracy
accuracy = accuracy_score(predictions, y_test)
print(f"Accuracy: {accuracy}")

model = AdaBoostClassifier()
model.fit(x_train, y_train)
preds = model.predict(x_test)
print(accuracy_score(preds, y_test))

model.fit(x_train, y_train)
preds = model.predict(x_test) 

print(classification_report(preds, y_test))

from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:, 1]
# Display evaluation metrics
print("Logistic Regression:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nRandom Forest:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Evaluate with confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ROC-AUC score
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc}")

# Display evaluation metrics
print("Logistic Regression:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nRandom Forest:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Random Forest (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

probs = model_lr.predict_proba(x_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, probs)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic Regression')
plt.legend()
plt.show()

# Calculate AUC
auc = roc_auc_score(y_test, probs)
print('AUC:', auc)