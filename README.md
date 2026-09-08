# Execution-CNN

A CNN for classifying trade execution outcomes from historical price data, using sliding-window feature engineering and focal loss to handle class imbalance.

## Overview

Predicts whether a trade entry is likely to lead to a favourable move within a short horizon. The pipeline downloads historical price/trade data, builds features via sliding windows, and trains a CNN to classify outcomes — evaluated not just on accuracy but on hit rate at high-confidence thresholds and Sharpe ratio, since raw classification accuracy doesn't say much about whether the signal is tradeable.

## Approach

- **Data**: `download_data.py` pulls historical price/trade data for a configurable set of tickers and date range.
- **Features**: `src/feature_engineering/engineering.py` builds fixed-length windows over the raw series.
- **Model**: `src/model/model.py` — a CNN trained with focal loss (to focus on the minority/informative class instead of oversampling with SMOTE), L2 regularisation, and Adam.
- **Evaluation**: `src/model/evaluate.py` reports classification metrics plus hit rate at high-confidence thresholds and Sharpe ratio on the held-out set.

## Project Structure

```
Execution-CNN/
├── download_data.py
├── main.py
├── src/
│   ├── feature_engineering/engineering.py
│   ├── model/{model,train,evaluate}.py
│   └── utils/
├── data/          # gitignored
└── test.ipynb     # exploration
```

## Results

| Metric | Value |
|---|---|
| Test accuracy | `[fill in]` |
| Hit rate @ confidence threshold | `[fill in]` |
| Sharpe ratio (held-out) | `[fill in]` |
| Tickers / date range | `[fill in]` |

## Setup

```bash
git clone https://github.com/nucleartoby/Execution-CNN.git
cd Execution-CNN
pip install -r requirements.txt
python download_data.py
python main.py
```

## Stack

Python, TensorFlow/Keras, NumPy, Pandas, scikit-learn.

## License

MIT