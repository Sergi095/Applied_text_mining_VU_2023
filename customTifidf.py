from sklearn.base import BaseEstimator,TransformerMixin

class custom_tfidf(BaseEstimator,TransformerMixin):
    def __init__(self,tfidf):
        self.tfidf = tfidf

    def fit(self, X, y=None):
        joined_X = X.apply(lambda x: ' '.join(x), axis=1)
        self.tfidf.fit(joined_X)        
        return self

    def transform(self, X):
        joined_X = X.apply(lambda x: ' '.join(x), axis=1)

        return self.tfidf.transform(joined_X)  