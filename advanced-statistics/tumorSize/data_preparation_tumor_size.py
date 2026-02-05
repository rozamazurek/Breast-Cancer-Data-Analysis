from sklearn.model_selection import train_test_split
import pandas as pd


def prepare_data():
    columns = ['Age', 'Race', 'Marital Status', 'T Stage', 'N Stage',
               '6th Stage', 'differentiate', 'Grade', 'A Stage',
               'Tumor Size', 'Estrogen Status', 'Progresterone Status',
               'Regional Node Examinated', 'Reginol Node Positive', 'Survival Months', 'Status']

    # Wczytywanie danych
    cancer_dataset = pd.read_csv("Breast_Cancer.csv", sep=',', header=None, names=columns)

    # Usuwanie nieprzydatnych danych: Race,Marital Status i Status
    cancer_dataset.drop(['Race', 'Marital Status', 'Status'], axis=1, inplace=True)\

    # Zamiana na wartośći numeryczne
    for col in ['Tumor Size', 'Age', 'Regional Node Examinated', 'Reginol Node Positive', 'Survival Months']:
        cancer_dataset[col] = pd.to_numeric(cancer_dataset[col], errors="coerce")

    # Usuwanie wierszy z jakimikolwiek brakami w danych - NaN
    cancer_dataset.dropna(inplace=True)

    # Podział na features i target
    x = cancer_dataset.drop("Tumor Size", axis=1)
    y = cancer_dataset["Tumor Size"]

    # Podział na kolumny numeryczne oraz kategoryczne
    cols_numerical = x.select_dtypes(include=['int64', 'float64']).columns
    cols_categorical = x.select_dtypes(include=['object', 'category']).columns

    # Podział danych na zbiór treningowy, testowy i walidacyjny
    x_train, x_test_temp, y_train, y_test_temp = train_test_split(x, y, test_size=0.3, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_test_temp, y_test_temp, test_size=0.5, random_state=42)

    return cols_numerical, cols_categorical, x_train, y_train, x_test, y_test, x_val, y_val
