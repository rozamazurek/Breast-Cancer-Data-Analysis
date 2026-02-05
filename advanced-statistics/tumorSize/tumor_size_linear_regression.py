import numpy as np
from data_preparation_tumor_size import prepare_data
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

# Transformacja na numpy
X_train_np = preprocessor.fit_transform(X_train)
X_test_np = preprocessor.transform(X_test)
X_val_np = preprocessor.transform(X_val)

# Dodanie bias- kolumny jedynek do cech
X_train_bias = np.c_[np.ones((X_train_np.shape[0], 1)), X_train_np]
X_test_bias = np.c_[np.ones((X_test_np.shape[0], 1)), X_test_np]
X_val_bias = np.c_[np.ones((X_test_np.shape[0], 1)), X_val_np]

#Konwersja na numpy- macierz (n,1)
y_train_np = y_train.values.reshape(-1, 1)
y_test_np = y_test.values.reshape(-1, 1)
y_val_np = y_val.values.reshape(-1, 1)

#  wzór na theta = (XᵀX)⁻¹Xᵀy
theta = np.linalg.inv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train_np

# Wzor pozwalajacy ominac macierze osobliwe
# theta = np.linalg.pinv(X_train_bias) @ y_train_np

# Predykcja wyników po wyliceniu theta z zamkniętej formuły
y_pred_train = X_train_bias @ theta
y_pred_test = X_test_bias @ theta
y_pred_val = X_val_bias @ theta

# Wyniki (wspólczynnik determinacji R-squared, Mean Absolute Error i Mean Squared Error)
print("\n Regresja liniowa (closed-form)")
print("R² score train:", r2_score(y_train_np, y_pred_train))
print("MAE train:", mean_absolute_error(y_train_np, y_pred_train))
print("MSE train:", mean_squared_error(y_train_np, y_pred_train))
print("R² score test:", r2_score(y_test_np, y_pred_test))
print("MAE test:", mean_absolute_error(y_test_np, y_pred_test))
print("MSE test:", mean_squared_error(y_test_np, y_pred_test))
print("R² score validation:", r2_score(y_val_np, y_pred_val))
print("MAE validation:", mean_absolute_error(y_val_np, y_pred_val))
print("MSE validation:", mean_squared_error(y_val_np, y_pred_val))
