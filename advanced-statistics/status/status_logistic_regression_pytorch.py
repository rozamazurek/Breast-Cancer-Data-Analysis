import time

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from data_preparation_status import prepare_data

# Przygotowywanie danych
cols_num, cols_cat, X_train, y_train, X_test, y_test = prepare_data()

# Pipeline dla cech numerycznych
transformer_numercial = Pipeline([('imputer', SimpleImputer(strategy='mean')),
                                  ('scaler', StandardScaler())])

# Pipeline dla cech numerycznych
transformer_categorical = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Połączenie obu powyżej
preprocessor = ColumnTransformer([
    ('num', transformer_numercial, cols_num),
    ('cat', transformer_categorical, cols_cat)
])

# Transformcja danych na numpy
X_train_np = preprocessor.fit_transform(X_train)
X_test_np = preprocessor.transform(X_test)

# Zmiana etykiet na liczby
mapping = {'Alive': 0, 'Dead': 1}
y_train_np = y_train.map(mapping).to_numpy().astype(np.int64)
y_test_np = y_test.map(mapping).to_numpy().astype(np.int64)

# Tworzenie dataset z numpy na tensory
train_ds = TensorDataset(torch.from_numpy(X_train_np).float(),
                         torch.from_numpy(y_train_np))
test_ds = TensorDataset(torch.from_numpy(X_test_np).float(),
                        torch.from_numpy(y_test_np))

# Ładowanie danych w batche, tasowanie w train
batch_size = 64
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size)


# Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# Trening
def train_clf(device):
    model = LogisticRegressionModel(X_train_np.shape[1]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # optymalizator do aktualizacji wag
    loss_fn = nn.BCELoss()  # funkcja straty - binary cross entropy loss
    epochs = 50
    train_losses = []
    train_accuracies = []
    start = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).float().unsqueeze(1)
            probs = model(xb)  # przewidywane prawdopodobieństwa klasy 1
            loss = loss_fn(probs, yb)  # funkcja straty
            optimizer.zero_grad()  # zerowanie gradientu
            loss.backward()  # obliczanie gradientu
            optimizer.step()  # aktualizacja wag
            total_loss += loss.item() * xb.size(0)

            preds = (probs > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(yb.cpu().numpy())

        avg_loss = total_loss / len(train_ds)
        y_pred = np.vstack(all_preds).ravel()
        y_true = np.vstack(all_labels).ravel()
        acc = accuracy_score(y_true, y_pred)

        train_losses.append(avg_loss)
        train_accuracies.append(acc)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} — loss: {avg_loss:.4f}, acc: {acc:.4f}")

    return model, time.time() - start


# Porównanie CPU i GPU
device_cpu = torch.device('cpu')
t_cpu = train_clf(device_cpu)[1]
if torch.cuda.is_available():
    device_gpu = torch.device('cuda')
    t_gpu = train_clf(device_gpu)[1]
else:
    t_gpu = None

print(f"Train time CPU: {t_cpu:.3f}s")
print(f"Train time GPU: {t_gpu:.3f}s" if t_gpu else "GPU not available")

# Ewaluacja
model_cpu, _ = train_clf(device_cpu)
model_cpu.eval()  # tryb ewaluacji
all_preds = []
all_labels = []
with torch.no_grad():  # wylaczamy gradienty
    for xb, yb in test_loader:
        xb = xb.to(device_cpu)
        probs = model_cpu(xb)
        preds = (probs > 0.5).cpu().numpy().ravel()
        all_preds.extend(preds)
        all_labels.extend(yb.numpy().ravel())

acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
rec = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f"\nTest Accuracy : {acc:.4f}")
print(f"Test Precision: {prec:.4f}")
print(f"Test Recall   : {rec:.4f}")
print(f"Test F1-score : {f1:.4f}")

if t_gpu:
    model_gpu, _ = train_clf(device_gpu)
    model_gpu.eval()
    all_preds_gpu = []
    all_labels_gpu = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device_gpu)
            probs = model_gpu(xb)
            preds = (probs > 0.5).cpu().numpy().ravel()
            all_preds_gpu.extend(preds)
            all_labels_gpu.extend(yb.numpy().ravel())

    acc_gpu = accuracy_score(all_labels_gpu, all_preds_gpu)
    prec_gpu = precision_score(all_labels_gpu, all_preds_gpu)
    rec_gpu = recall_score(all_labels_gpu, all_preds_gpu)
    f1_gpu = f1_score(all_labels_gpu, all_preds_gpu)

    print(f"\n[GPU] Test Accuracy : {acc_gpu:.4f}")
    print(f"[GPU] Test Precision: {prec_gpu:.4f}")
    print(f"[GPU] Test Recall   : {rec_gpu:.4f}")
    print(f"[GPU] Test F1-score : {f1_gpu:.4f}")