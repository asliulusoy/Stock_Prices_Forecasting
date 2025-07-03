import numpy as np
from sklearn.linear_model import LinearRegression

class FusionModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, lstm_preds, sentiment_scores, true_prices):
        X = np.column_stack([lstm_preds, sentiment_scores])
        y = np.array(true_prices)
        self.model.fit(X, y)

    def predict(self, lstm_pred, sentiment_score):
        return self.model.predict([[lstm_pred, sentiment_score]])[0]
