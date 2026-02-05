from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def set_params(scaler, cat_encoder, model, cols_numerical, cols_categorical):

    # Pipeline dla cech numerycznych
    transformer_numerical = Pipeline(steps=[
        ('num_trans', SimpleImputer(strategy='mean')),
        ('scaler', scaler)
    ])
    # Pipeline dla cech kategorycznych
    transformer_categorical = Pipeline(steps=[
        ('cat_trans', SimpleImputer(strategy='most_frequent')),
        ('onehot', cat_encoder)
    ])

    # Połączenie obu powyżej
    preprocessor = ColumnTransformer(transformers=[
        ('numerical', transformer_numerical, cols_numerical),
        ('categorical', transformer_categorical, cols_categorical)
    ])

    # Końcowy pipelne - preprocessing i model
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    return pipe
