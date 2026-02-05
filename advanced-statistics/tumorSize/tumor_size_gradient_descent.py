import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from data_preparation_tumor_size import prepare_data
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

# Przygotowanie danych
cols_numerical, cols_categorical, X_train, y_train, X_test, y_test, X_val, y_val = prepare_data()

# Pipeline dla cech numerycznych
transformer_numerical = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Pipeline dla cech kategorycznych
transformer_categorical = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Połączenie obu
preprocessor = ColumnTransformer(transformers=[
    ('num', transformer_numerical, cols_numerical),
    ('cat', transformer_categorical, cols_categorical)
])

# Transformacja danych
X_train_np = preprocessor.fit_transform(X_train)
X_test_np = preprocessor.transform(X_test)
X_val_np = preprocessor.transform(X_val)

# Dodanie biasu
X_train_bias = np.c_[np.ones((X_train_np.shape[0], 1)), X_train_np]
X_test_bias = np.c_[np.ones((X_test_np.shape[0], 1)), X_test_np]
X_val_bias = np.c_[np.ones((X_val_np.shape[0], 1)), X_val_np]

# Konwersja y
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)
y_val = y_val.values.reshape(-1, 1)

# Funkcja kosztu
def compute_cost(x, y, theta):
    m = x.shape[0]
    predictions = x @ theta
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    return cost

# Funkcja treningu modelu z regularyzacją
def train_model(X_train_bias, y_train, X_val_bias, y_val,
                regularization='none', lambda_=0.0,
                epochs=500, batch_size=64, learning_rate=0.01):
    
    m = X_train_bias.shape[0]
    theta = np.random.randn(X_train_bias.shape[1], 1)
    train_costs = []
    val_costs = []

    for epoch in range(epochs):
        permutation = np.random.permutation(m)
        X_shuffled = X_train_bias[permutation]
        y_shuffled = y_train[permutation]

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            y_pred = X_batch @ theta
            error = y_pred - y_batch
            gradient = (X_batch.T @ error) / X_batch.shape[0]

            if regularization == 'l2':
                reg_term = (lambda_ / X_batch.shape[0]) * theta
                reg_term[0] = 0
                gradient += reg_term
            elif regularization == 'l1':
                reg_term = (lambda_ / X_batch.shape[0]) * np.sign(theta)
                reg_term[0] = 0
                gradient += reg_term

            theta -= learning_rate * gradient

        train_costs.append(compute_cost(X_train_bias, y_train, theta))
        val_costs.append(compute_cost(X_val_bias, y_val, theta))

        if epoch % 100 == 0:
            r2_val = r2_score(y_val, X_val_bias @ theta)
            print(f"Epoch {epoch}: Train cost: {train_costs[-1]:.4f}, Validation cost: {val_costs[-1]:.4f}, R2: {r2_val:.4f}")

    return theta, train_costs, val_costs

# Trening modeli
theta_none, train_costs_none, val_costs_none = train_model(X_train_bias, y_train, X_val_bias, y_val, regularization='none', lambda_=0.0)
theta_l2, train_costs_l2, val_costs_l2 = train_model(X_train_bias, y_train, X_val_bias, y_val, regularization='l2', lambda_=0.1)
theta_l1, train_costs_l1, val_costs_l1 = train_model(X_train_bias, y_train, X_val_bias, y_val, regularization='l1', lambda_=0.1)

# Wyniki 
theta = theta_l2
y_pred_train = X_train_bias @ theta
y_pred_test = X_test_bias @ theta
y_pred_val = X_val_bias @ theta

print("\n Wyniki dla L2:")
print("R² score train:", r2_score(y_train, y_pred_train))
print("MAE train:", mean_absolute_error(y_train, y_pred_train))
print("MSE train:", mean_squared_error(y_train, y_pred_train))
print("R² score test:", r2_score(y_test, y_pred_test))
print("MAE test:", mean_absolute_error(y_test, y_pred_test))
print("MSE test:", mean_squared_error(y_test, y_pred_test))
print("R² score validation:", r2_score(y_val, y_pred_val))
print("MAE validation:", mean_absolute_error(y_val, y_pred_val))
print("MSE validation:", mean_squared_error(y_val, y_pred_val))

# Wykres funkcji kosztu
plt.figure(figsize=(10, 6))
plt.plot(range(len(train_costs_l2)), train_costs_l2, label='Train Cost (L2)')
plt.plot(range(len(val_costs_l2)), val_costs_l2, label='Validation Cost (L2)')
plt.xlabel("Epoch")
plt.ylabel("Cost (MSE)")
plt.title("Koszt podczas uczenia - Gradient Descent (L2)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Porównanie wag
weights_df = pd.DataFrame({
    'No Regularization': theta_none.flatten(),
    'L2 Regularization': theta_l2.flatten(),
    'L1 Regularization': theta_l1.flatten()
})

print("\nPorównanie pierwszych 10 wag:")
print(weights_df.head(10))

# Wykres porównania wag
plt.figure(figsize=(12, 6))
plt.plot(theta_none, label='No Regularization')
plt.plot(theta_l2, label='L2 Regularization')
plt.plot(theta_l1, label='L1 Regularization')
plt.title('Porównanie wag - brak regularyzacji vs L1 vs L2')
plt.xlabel('Indeks wagi')
plt.ylabel('Wartość wagi')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
