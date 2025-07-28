#  S&P 500 Market Prediction Using Machine Learning

This project analyzes and predicts the S&P 500 index movements using a combination of financial technical indicators and machine learning models. It features proper time-series cross-validation, walk-forward validation, and model-based strategy backtesting to simulate realistic market conditions.

---

##  Project Objectives

- Load, clean, and preprocess historical S&P 500 data
- Engineer technical indicators (RSI, MACD, Bollinger Bands, Momentum, etc.)
- Train machine learning models (Random Forest & XGBoost) to predict next-day market direction
- Evaluate models using TimeSeriesSplit and walk-forward validation
- Simulate trading strategies and assess profitability
- Analyze model performance across different market regimes

---

##  Features

-  Automatic data fetching from Yahoo Finance
-  Target creation based on next-day price movement
-  Feature engineering using `pandas_ta` technical indicators
-  Models: Random Forest & XGBoost
-  TimeSeriesSplit cross-validation
-  Walk-forward backtesting to avoid lookahead bias
-  Strategy simulation with capital growth visualization
-  Market regime analysis (bull, bear, high/low volatility)
-  Dashboard summary output and strategy performance plot

---

## Tech Stack

- Python
- Pandas, NumPy, Matplotlib, Seaborn
- `yfinance`, `pandas_ta`
- Scikit-learn, XGBoost
- tqdm, IPython display tools

---

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/sp500-ml-predictor.git
   cd sp500-ml-predictor

## Create a new virtual environment

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

## Install Requirements

pip install -r requirements.txt




## File Structure

sp500-ml-predictor/
│
├── project_v5(5).py              # Main project script
├── sp500.csv                     # Cached S&P 500 data (auto-created)
├── strategy_performance.png      # Final strategy plot
├── README.md                     # Project documentation
└── requirements.txt              # Python dependencies

## Usage

python project_v5(5).py
