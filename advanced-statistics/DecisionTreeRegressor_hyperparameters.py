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
from sklearn.tree import DecisionTreeRegressor

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
# Pipeline
pipe_tree = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor())
])

# Parametry
param_grid_tree = {
    'regressor__max_depth': [3, 5, 10, 15, None],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4],
    'regressor__max_features': [None, 'sqrt', 'log2']
}

grid_search_tree = GridSearchCV(pipe_tree, param_grid_tree, cv=3, scoring='r2', n_jobs=-1)
grid_search_tree.fit(X_train, y_train)

print("\nDecision Tree: Najlepsze parametry:", grid_search_tree.best_params_)
print("Najlepszy wynik R2 (cross-val):", grid_search_tree.best_score_)

# Test
y_test_pred_tree = grid_search_tree.best_estimator_.predict(X_test)
print("Test R2:", r2_score(y_test, y_test_pred_tree))
print("Test MAE:", mean_absolute_error(y_test, y_test_pred_tree))
print("Test MSE:", mean_squared_error(y_test, y_test_pred_tree))

# Heatmapa
cv_results_tree = pd.DataFrame(grid_search_tree.cv_results_)
pivot_tree = cv_results_tree.pivot_table(
    values='mean_test_score',
    index='param_regressor__max_depth',
    columns='param_regressor__min_samples_split'
)

plt.figure(figsize=(8, 6))
sns.heatmap(pivot_tree, annot=True, fmt=".3f", cmap="OrRd")
plt.title("R² Decision Tree – max_depth vs min_samples_split")
plt.xlabel("min_samples_split")
plt.ylabel("max_depth")
plt.tight_layout()
plt.show()