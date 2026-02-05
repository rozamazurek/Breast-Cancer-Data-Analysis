from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import numpy as np

class MixtureOfExperts(BaseEstimator, RegressorMixin):
    def __init__(self, experts, gater):
        self.experts = experts  # lista modeli ekspertów
        self.gater = gater      # model gatera 

    def fit(self, X, y):
        # Trenujemy ekspertów na całych danych
        self.experts_ = [clone(est).fit(X, y) for est in self.experts]
        
        # Predykcje ekspertów na treningu
        expert_preds = np.column_stack([est.predict(X) for est in self.experts_])
        
        y_array = np.array(y).reshape(-1, 1)  # konwersja do numpy i reshape
        best_expert = np.argmin(np.abs(expert_preds - y_array), axis=1)

 
        self.gater_ = clone(self.gater).fit(X, best_expert)
        return self
    
    def predict(self, X):
        # Predykcje ekspertów
        expert_preds = np.column_stack([est.predict(X) for est in self.experts_])
        
        # Prawdopodobieństwa przypisania do ekspertów
        weights = self.gater_.predict_proba(X)
        
        # Sumujemy predykcje ważone przez prawdopodobieństwa
        weighted_preds = np.sum(expert_preds * weights, axis=1)
        return weighted_preds
