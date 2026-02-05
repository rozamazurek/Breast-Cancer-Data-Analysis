import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from data_preparation_tumor_size import prepare_data

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
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Połączenie obu powyżej
preprocessor = ColumnTransformer(transformers=[
    ('numerical', transformer_numerical, cols_numerical),
    ('categorical', transformer_categorical, cols_categorical)
])

# Końcowy pipeline
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor())
])

# GridSearchCV - optymalizacja hiperparametrów
param_grid = {
     'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [3, 5],
    'regressor__learning_rate': [0.01, 0.1],
    'regressor__subsample': [0.6, 0.8, 1.0],
    'regressor__min_samples_split': [2, 5],
    'regressor__min_samples_leaf': [1, 3],
    'regressor__max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(pipe, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Najlepsze parametry:", grid_search.best_params_)
print("Najlepszy wynik R2 (cross-val):", grid_search.best_score_)

# Test końcowy z najlepszym modelem
best_model = grid_search.best_estimator_
y_test_pred = best_model.predict(X_test)

print("\n Test Set Results")
print("Test R2: ", r2_score(y_test, y_test_pred))
print("Test MAE:", mean_absolute_error(y_test, y_test_pred))
print("Test MSE:", mean_squared_error(y_test, y_test_pred))

# Wyniki GridSearchCV
cv_results = pd.DataFrame(grid_search.cv_results_)

cv_results['mean_test_score'] = cv_results['mean_test_score'].round(3)

# Pivot table
pivot = cv_results.pivot_table(
    values='mean_test_score',
    index='param_regressor__max_depth',
    columns='param_regressor__n_estimators'
)

plt.figure(figsize=(8, 6))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title("R² w zależności od max_depth i n_estimators")
plt.xlabel("n_estimators")
plt.ylabel("max_depth")
plt.tight_layout()
plt.show()