# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# File System management
import os
#import xgboost as xgb
#print(os.listdir("../input"))


# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder
# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
import seaborn as sns

_train = pd.read_csv('C:\\Users\\usaaxb85\\Documents\\Kaggle\\CreditDefaultRisk\\all\\application_train.csv')
print('Training Data Shape: ',_train.shape)
# Testing data features
_test = pd.read_csv('C:\\Users\\usaaxb85\\Documents\\Kaggle\\CreditDefaultRisk\\all\\application_test.csv')
print('Testing data shape: ', _test.shape)


# Function to calculate missing values by column# Funct
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()
    # print (mis_val)

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    # print (mis_val_percent)
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    # print (mis_val_table)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # print (mis_val_table_ren_columns.iloc[:,0])
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns

print(_train.dtypes.value_counts())
# Missing values statistics
missing_values = missing_values_table(_train)

# Number of unique classes in each object column
_train.select_dtypes(include =['object']).apply(pd.Series.nunique, axis = 0)

# Create a label encoder object
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in _train:
    if _train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(_train[col].unique())) <= 2:
            # Train on the training data
            le.fit(_train[col])
            # Transform both training and testing data
            _train[col] = le.transform(_train[col])
            _test[col] = le.transform(_test[col])

            # Keep track of how many columns were label encoded
            le_count += 1

print('%d columns were label encoded.' % le_count)

_train = pd.get_dummies(_train)
_test = pd.get_dummies(_test)

print('Training Features shape: ', _train.shape)
print('Testing Features shape: ', _test.shape)

train_labels = _train['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
_train, _test = _train.align(_test, join = 'inner', axis = 1)

# Add the target back in
_train['TARGET'] = train_labels

print('Training Features shape: ', _train.shape)
print('Testing Features shape: ', _test.shape)

print(_train.shape)
print(_test.shape)
_train.head(10)
_test.head(10)

# Create an anomalous flag column
_train['DAYS_EMPLOYED_ANOM'] = _train["DAYS_EMPLOYED"] == 365243

# Replace the anomalous values with nan
_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
plt.xlabel('Days Employment');

_test['DAYS_EMPLOYED_ANOM'] = _test["DAYS_EMPLOYED"] == 365243
_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

print('There are %d anomalies in the test data out of %d entries' % (_test["DAYS_EMPLOYED_ANOM"].sum(), len(_test)))



anom = _train[_train['DAYS_EMPLOYED'] == 365243]
non_anom = _train[_train['DAYS_EMPLOYED'] != 365243]
print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))
print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))
print('There are %d anomalous days of employment' % len(anom))

from sklearn.preprocessing import MinMaxScaler, Imputer

# Drop the target from the training data

features = list(_train.columns)
print(len(features))

if 'TARGET' in _train.columns:
    train = _train.drop(['TARGET'], axis = 1)
else:
    train = _train.copy()

# Feature names
features = list(train.columns)
print(len(features))

# Copy of the testing data
test = _test.copy()

# Median imputation of missing values
imputer = Imputer(strategy='median')

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range=(0, 1))
print(train.keys())
# Fit on the training data
imputer.fit(train)

# Transform both training and testing data
train = imputer.transform(train)
test = imputer.transform(_test)

# Repeat with the scaler
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)

from sklearn.linear_model import LogisticRegression

# Make the model with the specified regularization parameter
log_reg = LogisticRegression(C = 0.0001)

# Train on the training data
log_reg.fit(train, train_labels)

# Make predictions
# Make sure to select the second column only
log_reg_pred = log_reg.predict_proba(test)[:, 1]


# Submission dataframe
submit = _test[['SK_ID_CURR']]
submit['TARGET'] = log_reg_pred

submit.head()



# Save the submission to a csv file
submit.to_csv('log_reg_baseline.csv', index = False)



