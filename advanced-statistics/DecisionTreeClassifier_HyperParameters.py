from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import numpy as np
from data_preparation_6th_stage import prepare_data
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Przygotowywanie danych
cols_numerical, cols_categorical, X_train, y_train, X_test, y_test, X_val, y_val = prepare_data()

# Pipeline dla cech numerycznych
transformer_numerical = Pipeline(steps=[
    ('num_trans', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Pipeline dla cech kategorycznych
transformer_categorical = Pipeline(steps=[
    ('cat_trans', SimpleImputer(strategy='most_frequent')),
    ('onehot', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# Połączenie obu powyżej
preprocessor = ColumnTransformer(transformers=[
    ('numerical', transformer_numerical, cols_numerical),
    ('categorical', transformer_categorical, cols_categorical)
])
clf3 = DecisionTreeClassifier(random_state=42)

pipe_tree = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', clf3)
])

param_grid_tree = {
    'classifier__max_depth': [3, 5, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__criterion': ['gini', 'entropy']
}

grid_tree = GridSearchCV(pipe_tree, param_grid_tree, cv=3, scoring='accuracy', n_jobs=-1)
grid_tree.fit(X_train, y_train)

print("DecisionTreeClassifier - najlepsze parametry:", grid_tree.best_params_)
print("DecisionTreeClassifier - najlepsza dokładność (cross-val):", grid_tree.best_score_)

# Wyniki na teście
y_test_pred_tree = grid_tree.best_estimator_.predict(X_test)
print("\nDecisionTreeClassifier - Test Accuracy:", accuracy_score(y_test, y_test_pred_tree))
print("Precision:", precision_score(y_test, y_test_pred_tree, average='weighted', zero_division=0))
print("Recall:", recall_score(y_test, y_test_pred_tree, average='weighted', zero_division=0))
print("F1 Score:", f1_score(y_test, y_test_pred_tree, average='weighted', zero_division=0))

# Heatmapa 
results_tree = pd.DataFrame(grid_tree.cv_results_)
subset = results_tree[results_tree['param_classifier__criterion'] == 'gini']
pivot_tree = subset.pivot_table(
    values='mean_test_score',
    index='param_classifier__max_depth',
    columns='param_classifier__min_samples_split'
)

plt.figure(figsize=(8, 6))
sns.heatmap(pivot_tree, annot=True, fmt=".3f", cmap='Greens')
plt.title('Accuracy - DecisionTree (criterion=gini)')
plt.xlabel('min_samples_split')
plt.ylabel('max_depth')
plt.tight_layout()
plt.show()
