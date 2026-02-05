import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.svm import SVC
from data_preparation_6th_stage import prepare_data
from parameters import set_params

# Przygotowywanie danych
cols_numerical, cols_categorical, X_train, y_train, X_test, y_test, X_val, y_val = prepare_data()

# modele
classifiers = [
    LogisticRegression(max_iter=500),
    ExtraTreesClassifier(),
    RandomForestClassifier(),
    SVC()
]
# skalery
scalers = [StandardScaler(), MinMaxScaler()]

# kodery
encoders = [
    OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    OneHotEncoder(handle_unknown='ignore', sparse_output=False)
]

results = []

for model in classifiers:
    for num_scaler in scalers:
        for cat_encoder in encoders:
            # Tworzenie pipeline
            pipe = set_params(num_scaler, cat_encoder, model, cols_numerical, cols_categorical)

            # Trening
            pipe.fit(X_train, y_train)

            # Predykcja
            y_test_pred = pipe.predict(X_test)
            y_train_pred = pipe.predict(X_train)
            y_val_pred = pipe.predict(X_val)

            # Wyniki- dokładność, precyzja, czułość, f1-score średnia harmoniczna z precyzji i czułości

            results.append({
                'model': model.__class__.__name__,
                'num_scaler': num_scaler.__class__.__name__,

                'cat_encoder': cat_encoder.__class__.__name__,
                'accuracy_test': accuracy_score(y_test, y_test_pred),
                'precision_test': precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
                'recall_test': recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
                'f1_test': f1_score(y_test, y_test_pred, average='weighted', zero_division=0),

                'accuracy_train': accuracy_score(y_train, y_train_pred),
                'precision_train': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
                'recall_train': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
                'f1_train': f1_score(y_train, y_train_pred, average='weighted', zero_division=0),

                'accuracy_val': accuracy_score(y_val, y_val_pred),
                'precision_val': precision_score(y_val, y_val_pred, average='weighted', zero_division=0),
                'recall_val': recall_score(y_val, y_val_pred, average='weighted', zero_division=0),
                'f1_val': f1_score(y_val, y_val_pred, average='weighted', zero_division=0),
            })

# DataFrame z wynikami
pd.options.display.max_columns = None
results_df = pd.DataFrame(results)
print(results_df)
