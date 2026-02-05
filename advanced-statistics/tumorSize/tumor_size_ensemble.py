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
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Połączenie obu powyżej
preprocessor = ColumnTransformer(transformers=[
    ('numerical', transformer_numerical, cols_numerical),
    ('categorical', transformer_categorical, cols_categorical)
])

# Modele bazowe
reg1 = GradientBoostingRegressor(learning_rate=0.1,max_depth=3,max_features='log2',n_estimators=100,min_samples_leaf=1,min_samples_split=2,subsample=1.0)
reg2 = Ridge(alpha=1.0,solver='lsqr',fit_intercept=True)
reg3 = DecisionTreeRegressor(max_depth=3,min_samples_split=2,max_features=None,min_samples_leaf=4)

# Voting Regressor
voting_reg = VotingRegressor(estimators=[
    ('gbr', reg1),
    ('ridge', reg2),
    ('tree', reg3)
])

pipe_voting_reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', voting_reg)
])

pipe_voting_reg.fit(X_train, y_train)
y_pred_voting = pipe_voting_reg.predict(X_test)

print("\n VotingRegressor Results")
print("R2:", r2_score(y_test, y_pred_voting))
print("MAE:", mean_absolute_error(y_test, y_pred_voting))
print("MSE:", mean_squared_error(y_test, y_pred_voting))

# Stacking Regressor
stacking_reg = StackingRegressor(estimators=[
    ('gbr', reg1),
    ('ridge', reg2)
], final_estimator=GradientBoostingRegressor())

pipe_stacking_reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', stacking_reg)
])

pipe_stacking_reg.fit(X_train, y_train)
y_pred_stacking = pipe_stacking_reg.predict(X_test)

print("\n StackingRegressor Results")
print("R2:", r2_score(y_test, y_pred_stacking))
print("MAE:", mean_absolute_error(y_test, y_pred_stacking))
print("MSE:", mean_squared_error(y_test, y_pred_stacking))
