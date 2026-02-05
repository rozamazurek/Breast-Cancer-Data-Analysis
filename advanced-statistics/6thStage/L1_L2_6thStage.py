import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from data_preparation_6th_stage import prepare_data

# Dane
cols_numerical, cols_categorical, X_train, y_train, X_test, y_test, X_val, y_val = prepare_data()

# Pipeline dla danych
transformer_numerical = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

transformer_categorical = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer([
    ('num', transformer_numerical, cols_numerical),
    ('cat', transformer_categorical, cols_categorical)
])

# Modele do porównania
models = {
    "L2 Regularization (Ridge)": LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000),
    "L1 Regularization (Lasso)": LogisticRegression(penalty='l1', C=1.0, solver='liblinear', max_iter=1000),
}

# Test każdego modelu
for name, model in models.items():
    print(f"\n {name}")

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Trening
    pipe.fit(X_train, y_train)

    # Predykcja
    y_pred = pipe.predict(X_val)

    # Wyniki
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("Precision:", precision_score(y_val, y_pred, average='weighted', zero_division=0))
    print("Recall:", recall_score(y_val, y_pred, average='weighted', zero_division=0))
    print("F1 Score:", f1_score(y_val, y_pred, average='weighted', zero_division=0))

    # Wagi cech
    classifier = pipe.named_steps['classifier']
    if hasattr(classifier, 'coef_'):
        feature_names = pipe.named_steps['preprocessor'].get_feature_names_out()
        coefs = classifier.coef_[0]
        weights = pd.Series(coefs, index=feature_names)
        print("\n Top 10 cech wg absolutnej wagi:")
        print(weights.abs().sort_values(ascending=False).head(10))
