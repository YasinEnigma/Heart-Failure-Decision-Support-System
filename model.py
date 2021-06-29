import pandas as pd
import pandas_profiling as pp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os
import plotly.graph_objects as go
import plotly.io as pio
import pickle

from sklearn.utils import resample
# Metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, auc, roc_curve

# Validation
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline, make_pipeline

# Tuning
from sklearn.model_selection import GridSearchCV

# Fill Na Values
from sklearn.impute import SimpleImputer

# Feature Extraction
from sklearn.feature_selection import RFE

# Preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer, LabelEncoder

# Models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Ensembles
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier

warnings.filterwarnings('ignore')

sns.set_style("whitegrid", {'axes.grid': False})
pio.templates.default = "plotly_white"

dataset = pd.read_csv('processed.cleveland.csv', na_values=['?'])

# Analyze Data
def explore_data(dataset):
    print(f"Number of Instances: {dataset.shape[0]}\n")
    print(f"Number of Attributes: {dataset.shape[1]}\n")
    print(f"Dataset Columns: {dataset.columns}\n")
    print(f"Data types of each columns: {dataset.info()}\n")

# Checking for duplicates
def checking_removing_duplicates(dataset):
    count_dups = dataset.duplicated().sum()
    print("Number of Duplicates: ", count_dups)
    if count_dups >= 1:
        dataset.drop_duplicates(inplace=True)
        print('Duplicate values removed!')
    else:
        print('No Duplicate values')

# Checking for Null or NA values
def checking_removing_null_values(dataset):
    count_nulls = dataset.isna().sum()
    print("Number of Null Values: ", count_nulls)
    if count_nulls >= 1:
        dataset.dropna(axis=1, how='any', inplace=True)
        print('Null values removed!')
    else:
        print('No Null values')

# Fill Na value with mean strategy
def checking_fulling_null_values(dataset):
    count_nulls = dataset.isna().sum()
    print("Number of Null Values: ", count_nulls)
    if count_nulls >= 1:
        missingvalues = SimpleImputer(
            missing_values=np.nan, strategy='mean', axis=0)
        missingvalues = missingvalues.fit(dataset)
        dataset = missingvalues.transform(dataset)
        print('Null values Filled!')
    else:
        print('No Null values')

# Split training and validation set
def read_in_and_split_data(data, target):
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

# Spot-Check Algorithms
def GetModel():
    Models = []
    Models.append(('LR', LogisticRegression()))
    Models.append(('LDA', LinearDiscriminantAnalysis()))
    Models.append(('KNN', KNeighborsClassifier()))
    Models.append(('CART', DecisionTreeClassifier()))
    Models.append(('NB', GaussianNB()))
    Models.append(('SVM', SVC(probability=True)))
    Models.append(('MLP', MLPClassifier()))
    Models.append(('AB', AdaBoostClassifier()))
    Models.append(('GBM', GradientBoostingClassifier()))
    Models.append(('RF', RandomForestClassifier()))
    Models.append(('Bagging', BaggingClassifier()))
    Models.append(('ET', ExtraTreesClassifier()))
    return Models

# Normalizer Model
def Normalizer(nameOfScaler):
    if nameOfScaler == 'standard':
        scaler = StandardScaler()
    elif nameOfScaler == 'minmax':
        scaler = MinMaxScaler()
    elif nameOfScaler == 'normalizer':
        scaler = Normalizer()
    elif nameOfScaler == 'binarizer':
        scaler = Binarizer()

    pipelines = []
    pipelines.append(
        (nameOfScaler+'LR', Pipeline([('Scaler', scaler), ('LR', LogisticRegression())])))
    pipelines.append(
        (nameOfScaler+'LDA', Pipeline([('Scaler', scaler), ('LDA', LinearDiscriminantAnalysis())])))
    pipelines.append(
        (nameOfScaler+'KNN', Pipeline([('Scaler', scaler), ('KNN', KNeighborsClassifier())])))
    pipelines.append(
        (nameOfScaler+'CART', Pipeline([('Scaler', scaler), ('CART', DecisionTreeClassifier())])))
    pipelines.append(
        (nameOfScaler+'NB', Pipeline([('Scaler', scaler), ('NB', GaussianNB())])))
    pipelines.append(
        (nameOfScaler+'SVM', Pipeline([('Scaler', scaler), ('SVM', SVC())])))
    pipelines.append(
        (nameOfScaler+'MLP', Pipeline([('Scaler', scaler), ('MLP', MLPClassifier())])))
    pipelines.append(
        (nameOfScaler+'AB', Pipeline([('Scaler', scaler), ('AB', AdaBoostClassifier())])))
    pipelines.append(
        (nameOfScaler+'GBM', Pipeline([('Scaler', scaler), ('GMB', GradientBoostingClassifier())])))
    pipelines.append(
        (nameOfScaler+'RF', Pipeline([('Scaler', scaler), ('RF', RandomForestClassifier())])))
    pipelines.append(
        (nameOfScaler+'ET', Pipeline([('Scaler', scaler), ('ET', ExtraTreesClassifier())])))

    return pipelines

# Train model
def fit_model(X_train, y_train, models):
    # Test options and evaluation metric
    num_folds = 10
    scoring = 'accuracy'

    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds, random_state=0)
        cv_results = cross_val_score(
            model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    return names, results

# Save trained model
def save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))

# Performance Measure
def classification_metrics(model, conf_matrix):
    print(
        f"Training Accuracy Score: {model.score(X_train, y_train) * 100:.1f}%")
    print(
        f"Validation Accuracy Score: {model.score(X_test, y_test) * 100:.1f}%")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap='YlGnBu', fmt='g')
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.title('Confusion Matrix', fontsize=20, y=1.1)
    plt.ylabel('Actual label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.show()
    print(classification_report(y_test, y_pred))

# ROC_AUC
def roc_auc(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    print(f"roc_auc score: {auc(fpr, tpr)*100:.1f}%")
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=20)
    plt.legend()
    plt.show()


Q1 = dataset.quantile(0.25)
Q3 = dataset.quantile(0.75)
IQR = Q3 - Q1

dataset_out = dataset[~((dataset < (Q1 - 1.5 * IQR)) |
                        (dataset > (Q3 + 1.5 * IQR))).any(axis=1)]

health = dataset_out[dataset_out.target == 0]
disease = dataset_out[dataset_out.target == 1]

disease_upsampled = resample(disease,
                             replace=True,
                             n_samples=len(health),
                             random_state=0)

upsampled = pd.concat([health, disease_upsampled])


# Split data to training and validation set
target = 'target'
X_train, X_test, y_train, y_test = read_in_and_split_data(upsampled, target)

# Fit data to model
models = GetModel()
names, results = fit_model(X_train, y_train, models)

# Normalize data to improve accuracy
ScaledModel = Normalizer('minmax')
name, results = fit_model(X_train, y_train, ScaledModel)

# Fine tunong
model = GradientBoostingClassifier()
n_estimators = [10, 100, 1000]
learning_rate = [0.001, 0.01, 0.1]
subsample = [0.5, 0.7, 1.0]
max_depth = [3, 7, 9]

# define grid search
grid = dict(learning_rate=learning_rate, n_estimators=n_estimators,
            subsample=subsample, max_depth=max_depth)
kfold = KFold(n_splits=10, random_state=0)
cv_results = cross_val_score(
    model, X_train, y_train, cv=10, scoring='accuracy')
grid_search = GridSearchCV(estimator=model, param_grid=grid,
                           n_jobs=-1, cv=10, scoring='accuracy', error_score=0)
grid_result = grid_search.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# Predict unseen data
pipeline = make_pipeline(MinMaxScaler(),  GradientBoostingClassifier(
    learning_rate=0.1, max_depth=9, n_estimators=1000, subsample=0.7))
model = pipeline.fit(X_train, y_train)
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_metrics(pipeline, conf_matrix)
roc_auc(y_test, y_pred)

save_model(model, 'model.pkl')
