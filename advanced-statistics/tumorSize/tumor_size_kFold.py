import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder,PolynomialFeatures
from sklearn.linear_model import Ridge
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

# Końcowy pipelne - preprocessing i model
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor)
])

# Walidacja krzyżowa
kf = KFold(n_splits=3, shuffle=True, random_state=42)

fold_r2_scores = []
fold_mae_scores = []
fold_mse_scores = []

for train_idx, val_idx in kf.split(X_train):
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # Trening
    pipe.fit(X_train_fold, y_train_fold)

    # Predykcja
    y_pred = pipe.predict(X_val_fold)

    # Wyniki
    fold_r2_scores.append(r2_score(y_val_fold, y_pred))
    fold_mae_scores.append(mean_absolute_error(y_val_fold, y_pred))
    fold_mse_scores.append(mean_squared_error(y_val_fold, y_pred))

# DataFrame z wynikami
pd.options.display.max_columns = None
results_fold_r2 = pd.DataFrame(fold_r2_scores)
print(results_fold_r2)
results_fold_mae = pd.DataFrame(fold_mae_scores)
print(results_fold_mae)
results_fold_mse = pd.DataFrame(fold_mse_scores)
print(results_fold_mse)

results = [{
    'cv_r2_mean': np.mean(fold_r2_scores),
    'cv_mae_mean': np.mean(fold_mae_scores),
    'cv_mse_mean': np.mean(fold_mse_scores),
    'cv_r2_std': np.std(fold_r2_scores),
    'cv_mae_std': np.std(fold_mae_scores),
    'cv_mse_std': np.std(fold_mse_scores),
}]

# DataFrame z wynikami
pd.options.display.max_columns = None
results_df = pd.DataFrame(results)
print(results_df)


# Trening na całym zbiorze treningowym
pipe.fit(X_train, y_train)

# Predykcja na zbiorze testowym
y_test_pred = pipe.predict(X_test)

print("\n Test Set Results")
print("Test R2: ", r2_score(y_test, y_test_pred))
print("Test MAE:", mean_absolute_error(y_test, y_test_pred))
print("Test MSE:", mean_squared_error(y_test, y_test_pred))