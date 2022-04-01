import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
import matplotlib.pyplot as plot
# we can use the LabelEncoder to encode the gender feature
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# importing two different imputation methods that take into consideration all the features when predicting the missing values
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.dummy import DummyClassifier

# oversample the minority class using SMOTE
from imblearn.over_sampling import SMOTE
from collections import Counter

np.random.seed(42)

# load the dataset (1)
dataset = pd.read_csv('hcv_data_split.csv')
# print the dimensionality of the dataframe (1)
print('Dataframe shape: ', dataset.shape)
# print the names of the columns that can be used as features when training the machine learning model (1)
columns = list(dataset.columns)
# dropping non-features
columns.remove('split')
columns.remove('category')
print(columns)
# print the different data types that can be identified from the entire dataset (1)
print (dataset.dtypes)
# print the gender distribution in the complete dataset(i.e., the number of male and female individuals) (1)
print(dataset['Sex'].value_counts())
# print the class distribution of the entire dataset (1)
print(dataset['category'].value_counts())
# print the median age of patients in the dataset having the hepatitis C infection (1.5)
print("Median age for patients with hepatitis C:", dataset.loc[dataset['category'] == 1]['Age'].median())
# print the mean age of individuals in the dataset who does not have hepatitis C infection(i.e., the control group) (1.5)
print("Mean age", dataset.loc[dataset['category'] == 0]['Age'].mean())

# split the dataset into train and test based on the field "split" (0.5 + 0.5)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

for train_index, test_index in split.split(dataset, dataset['split']):
    train_set = dataset.loc[train_index]
    test_set = dataset.loc[test_index]

# print the dimensionality of the test dataset (0.5)
print("dimensionality of the test dataset", test_set.shape)

# print the dimensionality of the training dataset (0.5)
print("dimensionality of the training dataset", train_set.shape)

# print the proportional distribution of the classes to identify whether or not the classes are equally(or closer) distributed between the train and test datasets (1 + 1)
print("Test set, Class 0 distribution (%):", test_set.loc[test_set['category']==0].size / test_set.size * 100)
print("Test set, Class 1 distribution (%):", test_set.loc[test_set['category']==1].size / test_set.size * 100)
print("Train set, Class 0 distribution (%):", train_set.loc[train_set['category']==0].size / train_set.size * 100)
print("Train set, Class 1 distribution (%):", train_set.loc[train_set['category']==1].size / train_set.size * 100)
# Yes, they are equally distributed!

# analyze the distribution of the individual features(i.e., by using the complete dataset) and plot a feature that has a rough approximation of a Gaussian distribution (2)
# dataset.hist()
# By analysing the features, the feature that has the closest rough approximation og Gaussian distribution is the CHE feature
dataset["CHE"].hist()


# identify features that represent a notable correlation (i.e., either positive or negative correlation below or above -0.5 and 0.5) (3)
print("Features with notable correlation: ")
all_corr = dataset.corr()
for i in range(len(all_corr)):
    for j in range(i):
        if all_corr.iloc[i, j] > 0.5 or all_corr.iloc[i, j] < -0.5:
            print(str(all_corr.columns[i]) + " and " + str(all_corr.columns[j]) + " = " + str(all_corr.iloc[i, j]))


##### Model development (64/100)

# separate the features and the labels to be used in model development (2)
label = dataset["category"]
dataset = dataset.drop("category", axis=1)
train_set = train_set.drop("category", axis = 1)
test_set = test_set.drop("category", axis = 1)

# print the dimensionality of the dataset and the labels (0.5 + 0.5)
print("dimensionality of dataset: ", dataset.shape)
print("dimensionality of label: ", label.shape)
print("dimension of label: ", label.ndim)

# check for missing values in the training dataset and print how many rows can be identified with the missing values (1)
row_missing = []
for i in range(1, len(train_set)):
    if (train_set.iloc[i].isnull().values.any()):
        row_missing.append(train_set.iloc[i])
print("rows with missing values in train_set: ", len(row_missing))
    
# data imputation
# given the task in predicting individuals with hepatitis C infection, select two of the most appropriate imputation strategies to fill the missing values and briefly explain why you have selected the particular strategies in a markdown cell below the current cell (3)
imputer_simple = SimpleImputer(strategy='median') # Not selected
imputer_knn = KNNImputer(n_neighbors=5) # Selected
imputer_iter = IterativeImputer(max_iter=10) # Selected

# print the rows before and after being imputed with the two selected strategies (5)

#before using imputer, the Sex column needs to replace the string "m" and "f" into floats "1.0" and "0.0" respectively for the imputer to work
oE = OrdinalEncoder()
oE.fit(train_set[["Sex"]])
train_set[["Sex"]] = oE.transform(train_set[["Sex"]]) 

#print before imputed
print("Rows with missing values before being imputed:\n", train_set.iloc[missing_rows])

# KNN imputer
train_set_knn = pd.DataFrame(imputer_knn.fit_transform(train_set), columns=train_set.columns)
print("Rows with missing values with KNN imputer:\n", train_set_knn.iloc[missing_rows])

# Iterative imputer
train_set_iter = pd.DataFrame(imputer_iter.fit_transform(train_set), columns=train_set.columns)
print("Rows with missing values with Iterative imputer:\n", train_set_knn.iloc[missing_rows])

# indicate the encoding strategy that is more appropriate given the categorical feature 'Sex' and briefly explain why you selected one strategy over the other (i.e., either OrdinalEncoder or OneHotEncoder) in the markdown cell mentioned below (3)