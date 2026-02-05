import pandas as pd


def read_data():
    try:
        data = pd.read_csv("Breast_Cancer.csv")
        return data
    except FileNotFoundError:
        print("Błąd: Plik Breast_Cancer.csv nie został znaleziony.")
        return None
    except pd.errors.ParserError:
        print("Błąd: Problem z parsowaniem pliku CSV.")
        return None


def basic_statistics(data_set, csv_file_name):
    numeric_columns = data_set.select_dtypes(include=['number']).columns
    categorical_columns = data_set.select_dtypes(exclude=['number']).columns

    statistics = []

    for col in numeric_columns:
        statistics.append({
            "Feature": col,
            "Type": "Numerical",
            "Mean": data_set[col].mean(),
            "Median": data_set[col].median(),
            "Standard_Deviation": data_set[col].std(),
            "5th_Percentile": data_set[col].quantile(0.05),
            "95th_Percentile": data_set[col].quantile(0.95),
            "Missing_Values": data_set[col].isna().sum()
        })

    for col in categorical_columns:
        unique_values = data_set[col].nunique()
        missing_values = data_set[col].isna().sum()
        class_proportions = data_set[col].value_counts(normalize=True).to_dict()

        statistics.append({
            "Feature": col,
            "Type": "Categorical",
            "Unique_Classes": unique_values,
            "Missing_Values": missing_values,
            "Class_Proportions": "; ".join([f"{k}: {v:.2%}" for k, v in class_proportions.items()])
        })

    data_set_statistics = pd.DataFrame(statistics)
    data_set_statistics.to_csv(csv_file_name, index=False)
    return data_set_statistics
