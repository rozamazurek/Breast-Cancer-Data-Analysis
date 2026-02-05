from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def prepare_data():
    columns = ['Age', 'Race', 'Marital Status', 'T Stage', 'N Stage',
               '6th Stage', 'differentiate', 'Grade', 'A Stage',
               'Tumor Size', 'Estrogen Status', 'Progresterone Status',
               'Regional Node Examinated', 'Reginol Node Positive', 'Survival Months', 'Status']

    # Wczytywanie danych
    cancer_dataset = pd.read_csv("Breast_Cancer.csv", sep=',', header=None, names=columns)

    # Usuwanie nieprzydatnych danych: Age, Race, Marital Status, Status, Survival Months, Regional Node Examinated,
    # N Stage - duży wpływ na wynik, 100% accuracy
    cancer_dataset.drop(
        ['Age', 'Race', 'Marital Status', 'Status', 'Regional Node Examinated', 'Survival Months', 'N Stage'],
        axis=1, inplace=True)

    # Zamiana kolumn na wartośći numeryczne
    for col in ['Tumor Size', 'Reginol Node Positive']:
        cancer_dataset[col] = pd.to_numeric(cancer_dataset[col], errors="coerce")

    # Usuwanie wierszy z jakimikolwiek brakami w danych - NaN
    cancer_dataset.dropna(inplace=True)

    # Podział na features i target
    x = cancer_dataset.drop("6th Stage", axis=1)
    y_raw = cancer_dataset["6th Stage"]
    # Zamiana kategorii tekstowej [IIA, IIB, IIIA, IIIB, IIIC] na numeryczną [0,1,2,3,4]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    # fit uczy się jakie są unikalne etykiety a tranform przekształca je na liczby

    # Podział na kolumny numeryczne oraz kategoryczne
    cols_numerical = x.select_dtypes(include=['int64', 'float64']).columns
    cols_categorical = x.select_dtypes(include=['object', 'category']).columns

    # Podział danych na zbiór treningowy, testowy i walidacyjny
    x_train, x_test_temp, y_train, y_test_temp = train_test_split(x, y, test_size=0.3, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_test_temp, y_test_temp, test_size=0.5, random_state=42)

    return cols_numerical, cols_categorical, x_train, y_train, x_test, y_test, x_val, y_val
