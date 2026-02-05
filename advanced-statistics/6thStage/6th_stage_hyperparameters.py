import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import numpy as np
from data_preparation_6th_stage import prepare_data
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns

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

# Końcowy pipeline
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# GridSearchCV - optymalizacja hiperparametrów
param_grid_cls = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [5, 10, 20],
    'classifier__max_features': ['sqrt', 'log2']
}

grid_search_cls = GridSearchCV(pipe, param_grid_cls, cv=3, scoring='accuracy', n_jobs=-1)
grid_search_cls.fit(X_train, y_train)

print("Najlepsze parametry:", grid_search_cls.best_params_)
print("Najlepsza dokładność (cross-val):", grid_search_cls.best_score_)

# Test końcowy z najlepszym modelem
best_model = grid_search_cls.best_estimator_
y_test_pred = best_model.predict(X_test)

print("\n Test Set Results")
print("Test Accuracy: ", accuracy_score(y_test, y_test_pred))
print("Test precision:", precision_score(y_test, y_test_pred, average='weighted', zero_division=0))
print("Test recall:", recall_score(y_test, y_test_pred, average='weighted', zero_division=0))
print("Test f1:", f1_score(y_test, y_test_pred, average='weighted', zero_division=0))

# Heatmapa wyników z GridSearchCV
results_df = pd.DataFrame(grid_search_cls.cv_results_)
pivot_table = results_df.pivot_table(
    values='mean_test_score',
    index='param_classifier__max_depth',
    columns='param_classifier__n_estimators'
)

plt.figure(figsize=(8, 6))
sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap='Blues')
plt.title('Accuracy - GridSearchCV (RandomForestClassifier)')
plt.xlabel('n_estimators')
plt.ylabel('max_depth')
plt.tight_layout()
plt.show()