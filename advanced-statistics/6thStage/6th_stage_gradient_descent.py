from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from data_preparation_6th_stage import prepare_data
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np

# Przygotowywanie danych
cols_numerical, cols_categorical, X_train, y_train, X_test, y_test, X_val, y_val = prepare_data()

#zamiana na problem regresji logistycznej binarnej - IIIA vs reszta i i konwersja na numpy- macierz (n,1)
y_train = (y_train == 'IIIA').astype(int).reshape(-1, 1)
y_test = (y_test == 'IIIA').astype(int).reshape(-1, 1)
y_val = (y_val == 'IIIA').astype(int).reshape(-1, 1)

# Pipeline dla cech numerycznych
transformer_numerical = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Pipeline dla cech kategorycznych
transformer_categorical = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Połączenie obu powyżej
preprocessor = ColumnTransformer([
    ('num', transformer_numerical, cols_numerical),
    ('cat', transformer_categorical, cols_categorical)
])

# Transformacja na numpy
X_train_np = preprocessor.fit_transform(X_train)
X_test_np = preprocessor.transform(X_test)
X_val_np = preprocessor.transform(X_val)

# Dodanie bias- kolumny jedynek do cech
X_train_bias = np.c_[np.ones((X_train_np.shape[0], 1)), X_train_np]
X_test_bias = np.c_[np.ones((X_test_np.shape[0], 1)), X_test_np]
X_val_bias = np.c_[np.ones((X_val_np.shape[0], 1)), X_val_np]

# Funkcja sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Funkcja kosztu -
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon) #zabezpieczenie przed logarytmem z 0
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

#Parametry uczenia
learning_rate = 0.01
epochs = 500
batch_size = 64

# Inicjalizacja theta
theta = np.random.randn(X_train_bias.shape[1], 1)
m = X_train_bias.shape[0] # liczba próbek treningowych


# Gradient Descent
for epoch in range(epochs):
    permutation = np.random.permutation(m)
    X_shuffled = X_train_bias[permutation]
    y_shuffled = y_train[permutation]

    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i:i + batch_size] # bierzemy kawalek danych w rozmiarze batch size
        y_batch = y_shuffled[i:i + batch_size]

        # Przewidywania na danych w rozmiarze batch size
        z = X_batch @ theta # liniowa predykcja
        y_pred = sigmoid(z) # zamiana na prawdopodobienstwo z funkcja sigmoid
        error = y_pred - y_batch # Błąd predykcji

        # Gradient
        gradient = (X_batch.T @ error) / X_batch.shape[0]

        # Aktualizacja theta
        theta = theta - learning_rate * gradient

    # Funkcja kosztu i wyniki dla zbioru walidacyjnego
    if epoch % 50 == 0:
        val_preds = sigmoid(X_val_bias @ theta)
        val_preds_binary = (val_preds >= 0.5).astype(int)
        val_loss = binary_cross_entropy(y_val, val_preds)
        val_acc = accuracy_score(y_val, val_preds_binary)

        print(f"Epoch {epoch:>3}: Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

# Predykcja wyników
y_pred_train = sigmoid(X_train_bias @ theta) > 0.5
y_pred_test = sigmoid(X_test_bias @ theta) > 0.5
y_pred_val = sigmoid(X_val_bias @ theta) > 0.5

print("Train:")
print("Accuracy:", accuracy_score(y_train, y_pred_train))
print("Precision:", precision_score(y_train, y_pred_train))
print("Recall:", recall_score(y_train, y_pred_train))
print("F1 Score:", f1_score(y_train, y_pred_train))

print("\nTest:")
print("Accuracy:", accuracy_score(y_test, y_pred_test))
print("Precision:", precision_score(y_test, y_pred_test))
print("Recall:", recall_score(y_test, y_pred_test))
print("F1 Score:", f1_score(y_test, y_pred_test))

print("\nValidation:")
print("Accuracy:", accuracy_score(y_val, y_pred_val))
print("Precision:", precision_score(y_val, y_pred_val))
print("Recall:", recall_score(y_val, y_pred_val))
print("F1 Score:", f1_score(y_val, y_pred_val))
