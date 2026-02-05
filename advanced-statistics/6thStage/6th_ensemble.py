import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
from data_preparation_6th_stage import prepare_data
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

# Przygotowywanie danych
cols_numerical, cols_categorical, X_train, y_train, X_test, y_test, X_val, y_val = prepare_data()

# Pipeline dla cech numerycznych
transformer_numerical = Pipeline(steps=[
    ('num_trans', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Pipeline dla cech kategorycznych
transformer_categorical = Pipeline(steps=[
    ('cat_trans', SimpleImputer(strategy='most_frequent')),
    ('onehot', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# Połączenie obu powyżej
preprocessor = ColumnTransformer(transformers=[
    ('numerical', transformer_numerical, cols_numerical),
    ('categorical', transformer_categorical, cols_categorical)
])

# Bazowe modele dla ensemble
clf1 = RandomForestClassifier(random_state=42,max_depth=5,max_features='log2',n_estimators=200)
clf2 = HistGradientBoostingClassifier(random_state=42,learning_rate=0.01,max_depth=5,max_iter=200)
clf3 = DecisionTreeClassifier(random_state=42,criterion='gini',max_depth=5,min_samples_split=2)

# VotingClassifier (hard voting)
voting_clf = VotingClassifier(estimators=[
    ('rf', clf1),
    ('lr', clf2),
    ('dt', clf3)
], voting='hard')

pipe_voting = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', voting_clf)
])

pipe_voting.fit(X_train, y_train)
y_pred_voting = pipe_voting.predict(X_test)

print("\n VotingClassifier Results")
print("Accuracy:", accuracy_score(y_test, y_pred_voting))
print("Precision:", precision_score(y_test, y_pred_voting, average='weighted', zero_division=0))
print("Recall:", recall_score(y_test, y_pred_voting, average='weighted', zero_division=0))
print("F1:", f1_score(y_test, y_pred_voting, average='weighted', zero_division=0))

# StackingClassifier
stacking_clf = StackingClassifier(estimators=[
    ('rf', clf1),
    ('lr', clf2)
], final_estimator=LogisticRegression())

pipe_stacking = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', stacking_clf)
])

pipe_stacking.fit(X_train, y_train)
y_pred_stacking = pipe_stacking.predict(X_test)

print("\n StackingClassifier Results")
print("Accuracy:", accuracy_score(y_test, y_pred_stacking))
print("Precision:", precision_score(y_test, y_pred_stacking, average='weighted', zero_division=0))
print("Recall:", recall_score(y_test, y_pred_stacking, average='weighted', zero_division=0))
print("F1:", f1_score(y_test, y_pred_stacking, average='weighted', zero_division=0))
