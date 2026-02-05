import pandas as pd
from data_preparation_tumor_size import prepare_data
from parameters import set_params
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.svm import SVR
from sklearn.model_selection import KFold
import numpy as np

# Przygotowywanie danych
cols_numerical, cols_categorical, X_train, y_train, X_test, y_test, X_val, y_val = prepare_data()

# modele - regresory
estimators = [
    GradientBoostingRegressor(),
    RandomForestRegressor(),
    SVR()
]

# skalery
scalers = [StandardScaler(), MinMaxScaler()]

# kodery
encoders = [
    OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    OneHotEncoder(handle_unknown='ignore', sparse_output=False)
]

# Walidacja krzyżowa
kf = KFold(n_splits=3, shuffle=True, random_state=42)

results = []

for model in estimators:
    for num_scaler in scalers:
        for cat_encoder in encoders:

            fold_r2_scores = []
            fold_mae_scores = []
            fold_mse_scores = []

            for train_idx, val_idx in kf.split(X_train):
                X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

                # Tworzenie pipeline
                pipe = set_params(num_scaler, cat_encoder, model, cols_numerical, cols_categorical)

                # Trening
                pipe.fit(X_train_fold, y_train_fold)

                # Predykcja
                y_pred = pipe.predict(X_val_fold)

                # Wyniki
                fold_r2_scores.append(r2_score(y_val_fold, y_pred))
                fold_mae_scores.append(mean_absolute_error(y_val_fold, y_pred))
                fold_mse_scores.append(mean_squared_error(y_val_fold, y_pred))

            y_test_pred = pipe.predict(X_test)
            y_val_pred = pipe.predict(X_val)
            y_train_pred = pipe.predict(X_train)

            results.append({
                'model': model.__class__.__name__,
                'num_scaler': num_scaler.__class__.__name__,
                'cat_encoder': cat_encoder.__class__.__name__,
                'cv_r2_mean': np.mean(fold_r2_scores),
                'cv_mae_mean': np.mean(fold_mae_scores),
                'cv_mse_mean': np.mean(fold_mse_scores),
                'cv_r2_std': np.std(fold_r2_scores),
                'cv_mae_std': np.std(fold_mae_scores),
                'cv_mse_std': np.std(fold_mse_scores),
            })
            # Wyniki (wspólczynnik determinacji R-squared, Mean Absolute Error i Mean Squared Error)
            results.append({
                'model': model.__class__.__name__,
                'num_scaler': num_scaler.__class__.__name__,
                'cat_encoder': cat_encoder.__class__.__name__,
                'r2_test': r2_score(y_test, y_test_pred),
                'mae_test': mean_absolute_error(y_test, y_test_pred),
                'mse_test': mean_squared_error(y_test, y_test_pred),
                'r2_train': r2_score(y_train, y_train_pred),
                'mae_train': mean_absolute_error(y_train, y_train_pred),
                'mse_train': mean_squared_error(y_train, y_train_pred),
                'r2_val': r2_score(y_val, y_val_pred),
                'mae_val': mean_absolute_error(y_val, y_val_pred),
                'mse_val': mean_squared_error(y_val, y_val_pred),
            })

# DataFrame z wynikami
pd.options.display.max_columns = None
results_df = pd.DataFrame(results)
print(results_df)
