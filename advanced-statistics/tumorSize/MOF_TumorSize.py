from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from data_preparation_tumor_size import prepare_data
from Mixture_of_Experts import MixtureOfExperts

# Przygotowywanie danych
cols_numerical, cols_categorical, X_train, y_train, X_test, y_test, X_val, y_val = prepare_data()

# Przygotowanie pipeline 
transformer_numerical = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
transformer_categorical = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer([
    ('num', transformer_numerical, cols_numerical),
    ('cat', transformer_categorical, cols_categorical)
])

# Eksperci
expert1 = GradientBoostingRegressor(learning_rate=0.1,max_depth=3,max_features='log2',n_estimators=100,min_samples_leaf=1,min_samples_split=2,subsample=1.0)
expert2 = Ridge(alpha=1.0,solver='lsqr',fit_intercept=True)
expert3 = DecisionTreeRegressor(max_depth=3,min_samples_split=2,max_features=None,min_samples_leaf=4)

# Gater
from sklearn.linear_model import LogisticRegression
gater_model = LogisticRegression(multi_class='multinomial', max_iter=1000)

# Pipeline dla Mixture of Experts
moe = MixtureOfExperts(experts=[expert1, expert2, expert3], gater=gater_model)

pipe_moe = Pipeline([
    ('preprocessor', preprocessor),
    ('moe', moe)
])

pipe_moe.fit(X_train, y_train)
y_pred_moe = pipe_moe.predict(X_test)

print("\n Mixture of Experts Results")
print("R2:", r2_score(y_test, y_pred_moe))
print("MAE:", mean_absolute_error(y_test, y_pred_moe))
print("MSE:", mean_squared_error(y_test, y_pred_moe))
