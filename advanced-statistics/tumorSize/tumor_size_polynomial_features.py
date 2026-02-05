import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import Ridge

from data_preparation_tumor_size import prepare_data

# Przygotowywanie danych
cols_numerical, cols_categorical, X_train, y_train, X_test, y_test, X_val, y_val = prepare_data()

X_poly_train = X_train.copy()
y_poly_train = y_train.copy()
X_poly_val = X_val.copy()
y_poly_val = y_val.copy()

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

# Stopnie wielomianów do sprawdzenia
degrees = [1, 2, 3, 4]

train_results = []
val_results = []

for degree in degrees:
    # Końcowy pipelne - preprocessing i model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('regressor', Ridge(alpha=1.0))
    ])
    # Trening
    model.fit(X_poly_train, y_poly_train)

    # Predykcje
    y_train_pred = model.predict(X_poly_train)
    y_val_pred = model.predict(X_poly_val)

    # Wyniki
    train_results.append({
        "r2": r2_score(y_train, y_train_pred),
        "mae": mean_absolute_error(y_train, y_train_pred),
        "mse": mean_squared_error(y_train, y_train_pred)
    })

    val_results.append({
        "r2": r2_score(y_val, y_val_pred),
        "mae": mean_absolute_error(y_val, y_val_pred),
        "mse": mean_squared_error(y_val, y_val_pred)
    })


def plot_metric(metric_name, train_results, val_results):
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, [r[metric_name] for r in train_results], marker='o', label='Train')
    plt.plot(degrees, [r[metric_name] for r in val_results], marker='s', label='Validation')
    plt.title(f'Degree vs {metric_name.upper()}')
    plt.xlabel('Polynomial degree')
    plt.ylabel(metric_name.upper())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Wykresy
plot_metric("r2", train_results, val_results)
plot_metric("mae", train_results, val_results)
plot_metric("mse", train_results, val_results)
