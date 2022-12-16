from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import numpy as np


class BaseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features=None):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        return X_transformed

class KeepFeatures(BaseTransformer):

    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed = X[self.features]
        return X_transformed

class CustomLabelEncoder(BaseTransformer):

    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        self.transformations = defaultdict(dict)

        for f_i in self.features:
            le = LabelEncoder()
            X_transformed[f_i] = le.fit_transform(X_transformed[f_i])
            self.transformations[f_i] = dict(zip(le.classes_, list(range(len(le.classes_)))))

        return X_transformed

class AmountVsOldAndNewBalanceDest(BaseTransformer):
    def transform(self, X):
        assert "oldBalanceDest" in X.columns and "newBalanceDest" in X.columns,\
        f"Either {'oldBalanceDest'} or {'newBalanceDest'} columns are missing "

        X_transformed = X.copy()
        X_transformed.loc[
                (X_transformed.oldBalanceDest==0) & 
                (X_transformed.newBalanceDest==0) & 
                (X_transformed.amount!=0),
                ['oldBalanceDest','newBalanceDest']] = -1

        X_transformed['errorBalanceDest'] = X_transformed.oldBalanceDest + \
                                            X_transformed.amount - \
                                            X_transformed.newBalanceDest        
        return X_transformed

class AmountVsOldAndNewBalanceOrig(BaseTransformer):

    def transform(self, X):
        assert "oldBalanceOrig" in X.columns and "newBalanceOrig" in X.columns,\
        f"Either {'oldBalanceOrig'} or {'newBalanceOrig'} columns are missing "

        X_transformed = X.copy()
        X_transformed.loc[
                (X_transformed.oldBalanceOrig==0) & 
                (X_transformed.newBalanceOrig==0) & 
                (X_transformed.amount!=0),
                ['oldBalanceOrig','newBalanceOrig']] = np.nan

        X_transformed['errorBalanceOrig'] = X_transformed.newBalanceOrig + \
                                            X_transformed.amount - \
                                            X_transformed.oldBalanceOrig
        return X_transformed