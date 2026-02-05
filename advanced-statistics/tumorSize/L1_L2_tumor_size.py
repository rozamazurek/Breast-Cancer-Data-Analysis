import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from data_preparation_tumor_size import prepare_data

# Przygotowanie danych
cols_numerical, cols_categorical, X_train, y_train, X_test, y_test, X_val, y_val = prepare_data()

# Pipeline preprocessing
transformer_numerical = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

transformer_categorical = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', transformer_numerical, cols_numerical),
    ('cat', transformer_categorical, cols_categorical)
])

# Modele do przetestowania
models = {
    'No Regularization (LinearRegression)': LinearRegression(),
    'Ridge (L2, alpha=1.0)': Ridge(alpha=1.0),
    'Lasso (L1, alpha=0.1)': Lasso(alpha=0.1, max_iter=10000)
}

for name, model in models.items():
    print(f"\n {name}")

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    # Trening
    pipe.fit(X_train, y_train)

    # Predykcja
    y_pred = pipe.predict(X_val)

    # Wyniki
    print("Validation R²:", r2_score(y_val, y_pred))
    print("Validation MAE:", mean_absolute_error(y_val, y_pred))
    print("Validation MSE:", mean_squared_error(y_val, y_pred))

    # jezeli ridge lub lasso
    if hasattr(model, 'coef_'):
        # Wydobycie nazw cech po transformacjach
        features = pipe.named_steps['preprocessor'].get_feature_names_out()
        
        # Wydobycie współczynników
        regressor = pipe.named_steps['regressor']
        coefs = regressor.coef_ if regressor.coef_.ndim == 1 else regressor.coef_[0]
        weights = pd.Series(coefs, index=features)
        
        # Top 10 cech wg wartości absolutnej wagi
        top_weights = weights.abs().sort_values(ascending=False).head(10)
        print("Top 10 cech wg wartości wag:")
        print(top_weights)

        # Statystyki ogólne
        print(f"Liczba cech z wagą == 0: {(weights == 0).sum()}")
        print(f"Średnia wartość absolutna wag: {weights.abs().mean():.4f}")
        print(f"Maksymalna wartość wag: {weights.max():.4f}")
        print(f"Minimalna wartość wag: {weights.min():.4f}")
