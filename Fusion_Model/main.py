from helpers import get_ndx_prices, generate_lstm_predictions, collect_sentiment_series, build_fusion_training_data
from Fusion_Model.fuse import FusionModel

start_date = "2025-03-10"
end_date = "2025-05-31"
api_key = "-"

print("Fetching data...")
prices = get_ndx_prices(start_date, end_date)
lstm_preds = generate_lstm_predictions(start_date, end_date)
sentiment_scores = collect_sentiment_series(start_date, end_date, api_key)

print("Building training data...")
lstm_preds_train, sentiment_scores_train, true_prices_train = build_fusion_training_data(
    prices, lstm_preds, sentiment_scores
)

print("Training fusion model...")
fusion = FusionModel()
fusion.train(lstm_preds_train, sentiment_scores_train, true_prices_train)

print("Evaluating...")
example_pred = fusion.predict(lstm_preds_train[-1], sentiment_scores_train[-1])
print("Predicted final price:", example_pred)