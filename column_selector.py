from sklearn.base import TransformerMixin, BaseEstimator


class TextColumnSelector(BaseEstimator, TransformerMixin):
    # Transformer to select a single column from the data frame to perform additional transformations.
    # This class will select columns containing text data.

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]

    # def fit_transform(self, X, y=None):
    #     self.fit(X, y)
    #     return self.transform(X)
    #
    # def __getitem__(self, x):
    #     return self.X[x], self.y[x]


class NumColumnSelector(BaseEstimator, TransformerMixin):
    # Transformer to select a single column from the data frame to perform additional transformations.
    # This class will select the columns containing numeric data.

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]


class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()