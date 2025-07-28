#!/usr/bin/env python
# coding: utf-8

# In[103]:


import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm


# In[105]:


def load_data():
    """Load S&P 500 data from file or download if not available"""
    try:
        sp500 = pd.read_csv("sp500.csv", index_col=0)
        if sp500.empty:
            raise FileNotFoundError
        # Convert index to datetime with consistent timezone handling
        sp500.index = pd.to_datetime(sp500.index, utc=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("Downloading S&P 500 data...")
        sp500 = yf.Ticker("^GSPC")
        sp500 = sp500.history(period="max")
        # Save with consistent timezone info
        sp500.to_csv("sp500.csv")
        # Ensure index has consistent timezone
        if sp500.index.tz is not None:
            sp500.index = sp500.index.tz_convert('UTC')
        else:
            sp500.index = sp500.index.tz_localize('UTC')

    return sp500


# In[107]:


def prepare_data(data):
    """Clean and prepare the base dataset"""
    data = data.copy()
    data.drop(columns=["Dividends", "Stock Splits"], inplace=True, errors="ignore")

    # Create target variable (next day's movement)
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)

    # Make sure start_date is timezone-aware if the data index is
    start_date = pd.to_datetime("1990-01-01")
    if data.index.tz is not None:
        start_date = start_date.tz_localize(data.index.tz)
    
    # Find the first date that's >= start_date
    if start_date not in data.index:
        valid_dates = data.index[data.index >= start_date]
        if len(valid_dates) > 0:
            start_date = valid_dates[0]
            print(f"Warning: 1990-01-01 not found, using {start_date.date()} instead")
        else:
            # Handle case where no valid dates are found
            print("No dates after 1990-01-01 found in the dataset")
            return data.copy()  # Return all data if no valid date found

    return data.loc[start_date:].copy()


# In[109]:


def add_features(df):
    """Add technical indicators and features to the dataframe, with error handling for short input"""
    result_df = df.copy()

    # Early exit if not enough data to compute most indicators
    min_rows_required = 50  # For safety: MACD (26), Bollinger (20), Up_Days (50)
    if len(result_df) < min_rows_required:
        print("Not enough rows to compute features, returning partial or empty feature set.")
        return result_df

    # RSI
    rsi = ta.rsi(result_df['Close'], length=14)
    result_df['RSI_14'] = rsi if rsi is not None else np.nan

    # MACD
    macd = ta.macd(result_df['Close'], fast=12, slow=26, signal=9)
    if macd is not None:
        result_df['MACD'] = macd.get('MACD_12_26_9', np.nan)
        result_df['MACD_Signal'] = macd.get('MACDs_12_26_9', np.nan)
        result_df['MACD_Hist'] = macd.get('MACDh_12_26_9', np.nan)
    else:
        result_df['MACD'] = np.nan
        result_df['MACD_Signal'] = np.nan
        result_df['MACD_Hist'] = np.nan

    # Bollinger Bands
    bbands = ta.bbands(result_df['Close'], length=20)
    if bbands is not None:
        result_df['Upper_Band'] = bbands.get('BBU_20_2.0', np.nan)
        result_df['Middle_Band'] = bbands.get('BBM_20_2.0', np.nan)
        result_df['Lower_Band'] = bbands.get('BBL_20_2.0', np.nan)
    else:
        result_df['Upper_Band'] = np.nan
        result_df['Middle_Band'] = np.nan
        result_df['Lower_Band'] = np.nan

    # Price relative to moving averages
    horizons = [5, 20, 50, 200]
    for horizon in horizons:
        ma = result_df['Close'].rolling(window=horizon).mean()
        result_df[f'MA_{horizon}'] = ma
        result_df[f'Close_to_MA_{horizon}'] = result_df['Close'] / ma - 1

    # Volatility
    result_df['Volatility_20'] = result_df['Close'].pct_change().rolling(window=20).std()

    # Momentum
    result_df['5_day_momentum'] = result_df['Close'].pct_change(periods=5)
    result_df['10_day_momentum'] = result_df['Close'].pct_change(periods=10)

    # Previous market direction
    result_df['Prev_day_return'] = result_df['Close'].pct_change()

    # Market trend (% of up days in last periods)
    for period in [10, 20, 50]:
        result_df[f'Up_days_{period}'] = (
            (result_df['Close'] > result_df['Close'].shift(1))
            .rolling(period)
            .mean()
        )

    return result_df


# In[111]:


def add_features_no_lookahead(df, current_idx):
    """Add features using only data available up to current_idx to prevent lookahead bias"""
    # Work only with data up to current_idx
    temp_df = df.iloc[:current_idx].copy()
    
    # Add features to the temporary dataframe
    temp_df_with_features = add_features(temp_df)
    
    # Return only the latest row with features
    return temp_df_with_features.iloc[-1:]


# In[113]:


def select_features():
    """Define the feature set to use for prediction"""
    features = [
        'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'Close_to_MA_5', 'Close_to_MA_20', 'Close_to_MA_50', 'Close_to_MA_200',
        'Volatility_20', '5_day_momentum', '10_day_momentum',
        'Prev_day_return', 'Up_days_10', 'Up_days_20', 'Up_days_50'
    ]
    return features


# In[115]:


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Evaluate model with multiple metrics"""
    probas = model.predict_proba(X_test)[:, 1]
    preds = (probas >= threshold).astype(int)
    
    results = {
        'accuracy': (preds == y_test).mean(),
        'precision': precision_score(y_test, preds),
        'roc_auc': roc_auc_score(y_test, probas),
        'classification_report': classification_report(y_test, preds)
    }
    
    return results, preds, probas


# In[117]:


def proper_time_series_cv(data, features, n_splits=5):
    """Perform proper time-series cross-validation"""
    # First, add features to the dataset
    data_with_features = add_features(data)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Initialize models with regularization to prevent overfitting
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        min_samples_split=50,
        min_samples_leaf=20,
        max_features='sqrt',
        random_state=42
    )
    
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42
    )
    
    models = {
        'RandomForest': rf_model,
        'XGBoost': xgb_model
    }
    
    results = {}
    feature_importances = {}
    
    # Drop any rows with NaN in features
    valid_data = data_with_features.dropna(subset=features)
    
    X = valid_data[features]
    y = valid_data['Target']
    
    for model_name, model in models.items():
        cv_metrics = {
            'accuracy': [],
            'precision': [],
            'roc_auc': []
        }
        all_preds = []
        all_true = []
        
        print(f"\n--- Running {model_name} with Time Series CV ---")
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Evaluate
            eval_results, preds, _ = evaluate_model(model, X_test, y_test)
            
            # Store metrics
            for metric in ['accuracy', 'precision', 'roc_auc']:
                cv_metrics[metric].append(eval_results[metric])
            
            all_preds.extend(preds)
            all_true.extend(y_test.values)
            
            print(f"Fold results - Accuracy: {eval_results['accuracy']:.4f}, "
                  f"Precision: {eval_results['precision']:.4f}, "
                  f"ROC AUC: {eval_results['roc_auc']:.4f}")
        
        # Average CV metrics
        avg_metrics = {k: np.mean(v) for k, v in cv_metrics.items()}
        
        # Store feature importances
        if model_name == 'RandomForest':
            feature_importances[model_name] = dict(zip(features, model.feature_importances_))
        elif model_name == 'XGBoost':
            feature_importances[model_name] = dict(zip(features, model.feature_importances_))
        
        # Overall model performance
        results[model_name] = {
            'cv_metrics': avg_metrics,
            'overall_precision': precision_score(all_true, all_preds),
            'overall_classification_report': classification_report(all_true, all_preds)
        }
        
        print(f"\n{model_name} CV Average - Accuracy: {avg_metrics['accuracy']:.4f}, "
              f"Precision: {avg_metrics['precision']:.4f}, "
              f"ROC AUC: {avg_metrics['roc_auc']:.4f}")
        print(f"Overall Classification Report:\n{results[model_name]['overall_classification_report']}")
    
    return results, feature_importances, models


# In[119]:


def walk_forward_validation(data, features, train_size=1000, test_size=250):
    """
    Perform walk-forward validation (more realistic backtest)
    This prevents look-ahead bias by only using available data at each point
    """
    # First, let's make sure we have the basic target column
    if 'Target' not in data.columns:
        data = prepare_data(data)
    
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100, 
            min_samples_split=50,
            min_samples_leaf=20,
            max_features='sqrt',
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            learning_rate=0.01,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42
        )
    }
    
    # We'll store predictions and true values here
    predictions = {model_name: [] for model_name in models}
    true_values = []
    dates = []
    
    # Need enough data to compute initial features
    min_required = 200  # Arbitrary but must be at least max of our lookback periods
    
    # Start training after we have enough data
    first_train_end = min_required + train_size
    
    if first_train_end >= len(data):
        print("Not enough data for walk-forward validation")
        return None
    
    # Walk forward through time
    for test_end in range(first_train_end + test_size, len(data), test_size):
        test_start = test_end - test_size
        train_start = test_start - train_size
        
        if train_start < min_required:
            train_start = min_required
            
        print(f"\nTraining on data from index {train_start} to {test_start-1}")
        print(f"Testing on data from index {test_start} to {test_end-1}")
        
        # Get dates for this test period for reference
        test_dates = data.index[test_start:test_end]
        print(f"Date range: {test_dates[0].date()} to {test_dates[-1].date()}")
        
        # Get training data and add features
        train_data = data.iloc[train_start:test_start].copy()
        train_data_with_features = add_features(train_data)
        
        # Extract features and target for training
        X_train = train_data_with_features[features].copy()
        y_train = train_data_with_features['Target'].copy()
        
        # Handle NaN values
        X_train = X_train.dropna()
        y_train = y_train.loc[X_train.index]
        
        # Train each model
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            
            # Now prepare test data with features based on information available at each point
            all_test_preds = []
            
            # Go through each test day one by one to avoid lookahead bias
            for i in tqdm(range(test_start, test_end), desc=f"Testing {model_name}"):
                # Get features up to current point (only use data available at time i)
                point_data = data.iloc[:i+1].copy()
                point_data_with_features = add_features(point_data)
                point_features = point_data_with_features.iloc[-1:][features]
                
                # Skip if any features are NaN
                if point_features.isna().any(axis=1).iloc[0]:
                    # Change: Use float NaN not None
                    all_test_preds.append(np.nan)
                    continue
                
                # Make prediction for this single day
                point_pred = model.predict_proba(point_features)[0, 1]
                # Change: Store as float instead of binary classification
                all_test_preds.append(float(1 if point_pred >= 0.5 else 0))
            
            # Store predictions for this test period
            predictions[model_name].extend(all_test_preds)
        
        # Store true values and dates for this test period
        true_values.extend(data.iloc[test_start:test_end]['Target'].values)
        dates.extend(data.index[test_start:test_end])
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Date': dates,
        'True': true_values
    })
    
    # Add predictions from each model
    for model_name, preds in predictions.items():
        # Make sure predictions are numeric
        results_df[f'{model_name}_Pred'] = pd.to_numeric(preds, errors='coerce')
    
    results_df.set_index('Date', inplace=True)
    
    # Add market data to results
    results_df['Close'] = data.loc[results_df.index, 'Close']
    
    # Calculate performance metrics for each model
    model_metrics = {}
    for model_name in models:
        # Skip NaN values
        valid_idx = results_df[f'{model_name}_Pred'].notna()
        if valid_idx.sum() > 0:
            y_true = results_df.loc[valid_idx, 'True']
            y_pred = results_df.loc[valid_idx, f'{model_name}_Pred']
            
            model_metrics[model_name] = {
                'precision': precision_score(y_true, y_pred),
                'accuracy': (y_true == y_pred).mean(),
                'classification_report': classification_report(y_true, y_pred)
            }
            
            print(f"\n--- {model_name} Walk-Forward Results ---")
            print(f"Precision: {model_metrics[model_name]['precision']:.4f}")
            print(f"Accuracy: {model_metrics[model_name]['accuracy']:.4f}")
            print(f"Classification Report:\n{model_metrics[model_name]['classification_report']}")

        # Add clear predictions to results_df
    for model_name in models:
        results_df[f'{model_name}_Clear_Pred'] = results_df[f'{model_name}_Pred'].map(
            lambda x: "Up" if x == 1 else ("Down" if x == 0 else np.nan)
        )
    
    return results_df, model_metrics


# In[121]:


def calculate_returns(predictions_df, initial_balance=10000):
    """Calculate strategy returns without lookahead bias"""
    results = predictions_df.copy()
    
    # Calculate daily returns of underlying asset
    results['Market_Return'] = results['Close'].pct_change()
    
    # Calculate strategy returns for each model
    model_columns = [col for col in results.columns if col.endswith('_Pred')]
    
    for model_col in model_columns:
        model_name = model_col.replace('_Pred', '')
        
        # Ensure all values are numeric before multiplication
        # This is the key fix for the error
        shifted_preds = pd.to_numeric(results[model_col].shift(1), errors='coerce')
        
        # Strategy return: previous day's prediction * current day's return
        # This avoids lookahead bias
        results[f'{model_name}_Return'] = results['Market_Return'] * shifted_preds
        
        # Calculate cumulative returns
        results[f'{model_name}_Value'] = (1 + results[f'{model_name}_Return']).cumprod().fillna(1) * initial_balance
    
    # Buy and hold strategy
    results['Buy_Hold_Value'] = (1 + results['Market_Return']).cumprod().fillna(1) * initial_balance
    
    return results


# In[123]:


def analyze_market_regimes(returns_df):
    """Analyze performance in different market regimes"""
    # Define market regimes
    returns_df['Rolling_Volatility'] = returns_df['Market_Return'].rolling(20).std()
    returns_df['Rolling_Return'] = returns_df['Market_Return'].rolling(60).sum()
    
    # Define regimes
    regimes = {
        'Bull_Market': returns_df['Rolling_Return'] > 0.05,
        'Bear_Market': returns_df['Rolling_Return'] < -0.05,
        'High_Volatility': returns_df['Rolling_Volatility'] > returns_df['Rolling_Volatility'].quantile(0.8),
        'Low_Volatility': returns_df['Rolling_Volatility'] < returns_df['Rolling_Volatility'].quantile(0.2)
    }
    
    # Analyze each model in each regime
    model_columns = [col for col in returns_df.columns if col.endswith('_Return') and not col == 'Market_Return']
    
    regime_results = {}
    
    for regime_name, regime_mask in regimes.items():
        regime_results[regime_name] = {}
        
        for model_col in model_columns:
            model_name = model_col.replace('_Return', '')
            
            # Get returns for this model in this regime
            regime_returns = returns_df.loc[regime_mask, model_col].dropna()
            
            if len(regime_returns) > 0:
                regime_results[regime_name][model_name] = {
                    'mean_return': regime_returns.mean(),
                    'win_rate': (regime_returns > 0).mean(),
                    'total_return': (1 + regime_returns).prod() - 1,
                    'count': len(regime_returns)
                }
    
    return regime_results


# In[125]:


def plot_results(returns_df):
    """Plot strategy performance and analysis"""
    model_value_cols = [col for col in returns_df.columns if col.endswith('_Value')]
    
    plt.figure(figsize=(14, 8))
    
    # Plot equity curves
    for col in model_value_cols + ['Buy_Hold_Value']:
        plt.plot(returns_df[col].dropna(), label=col.replace('_Value', ''))
    
    plt.title('Strategy Performance Comparison')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('strategy_performance.png')
    plt.show()


# In[127]:


def get_clear_prediction(model, X, threshold=0.5):
    """
    Get clear "Up" or "Down" prediction for tomorrow's market direction
    Returns:
        - "Up" if predicted probability >= threshold
        - "Down" if predicted probability < threshold
    """
    proba = model.predict_proba(X)[:, 1][0]  # Get probability of upward movement
    return "Up" if proba >= threshold else "Down"


# In[143]:


def display_dashboard_summary(returns_df, regime_analysis, model_metrics, latest_predictions, initial_balance=10000):
    """
    Display a dashboard-style summary of model performance.
    Args:
        - returns_df: DataFrame from calculate_returns()
        - regime_analysis: output from analyze_market_regimes()
        - model_metrics: output from walk_forward_validation()
        - latest_predictions: dict of latest "Up"/"Down" predictions per model
        - initial_balance: starting portfolio value
    """
    import pandas as pd
    from IPython.display import display

    print("\n📊 === DASHBOARD SUMMARY === 📊\n")

    # Final portfolio values
    print("💰 Final Strategy Portfolio Values:")
    summary = {}
    for col in returns_df.columns:
        if col.endswith("_Value") and col != "Buy_Hold_Value":
            model_name = col.replace("_Value", "")
            final_value = returns_df[col].dropna().iloc[-1]
            summary[model_name] = {"Final Value ($)": f"{final_value:,.2f}"}

    # Add Buy & Hold
    final_bh = returns_df["Buy_Hold_Value"].dropna().iloc[-1]
    summary["BuyHold"] = {"Final Value ($)": f"{final_bh:,.2f}"}

    # Add metrics
    for model, metrics in model_metrics.items():
        summary[model].update({
            "Precision": f"{metrics['precision']:.4f}",
            "Accuracy": f"{metrics['accuracy']:.4f}"
        })

    # Add latest predictions
    for model, pred in latest_predictions.items():
        summary[model]["Latest Prediction"] = pred

    # Display main summary
    summary_df = pd.DataFrame(summary).T
    display(summary_df.style.set_caption("Strategy Performance Summary"))

    # Market Regime Summary
    print("\n📈 Market Regime Performance (mean return % and win rate):")
    regime_summary = []
    for regime, model_stats in regime_analysis.items():
        for model, metrics in model_stats.items():
            regime_summary.append({
                "Regime": regime,
                "Model": model,
                "Mean Return (%)": round(metrics['mean_return'] * 100, 4),
                "Win Rate (%)": round(metrics['win_rate'] * 100, 2),
                "Total Return (%)": round(metrics['total_return'] * 100, 2),
                "Samples": metrics['count']
            })
    regime_df = pd.DataFrame(regime_summary)
    display(regime_df.pivot(index="Model", columns="Regime", values="Mean Return (%)")
            .style.set_caption("Mean Return (%) by Market Regime"))


# In[145]:


def main():
    # Load and prepare data
    print("Loading and preparing data...")
    sp500 = load_data()
    sp500 = prepare_data(sp500)

    # Define features to use
    features = select_features()
    print(f"Selected features: {features}")

    # Time-series cross-validation
    print("\n=== Running Time Series Cross-Validation ===")
    cv_results, feature_importances, trained_models = proper_time_series_cv(sp500, features)

    # Walk-forward validation (realistic backtest)
    print("\n=== Running Walk-Forward Validation ===")
    wf_results_df, wf_metrics = walk_forward_validation(sp500, features)

    if wf_results_df is not None:
        # Strategy returns
        print("\n=== Calculating Strategy Returns ===")
        returns_df = calculate_returns(wf_results_df)

        # Market regime analysis
        print("\n=== Analyzing Market Regimes ===")
        regime_analysis = analyze_market_regimes(returns_df)

        for regime, models in regime_analysis.items():
            print(f"\n--- {regime} Performance ---")
            for model, metrics in models.items():
                print(f"{model}: Win Rate: {metrics['win_rate']:.4f}, "
                      f"Mean Return: {metrics['mean_return']*100:.4f}%, "
                      f"Total Return: {metrics['total_return']*100:.2f}% "
                      f"(Sample size: {metrics['count']})")

        # === Latest Market Prediction ===
        print("\n=== Latest Market Prediction ===")
        lookback_window = 250  # Lookback days for latest indicators
        latest_data = sp500.iloc[-lookback_window:].copy()
        latest_with_features = add_features(latest_data)

        # Drop rows with missing values for selected features
        latest_with_features = latest_with_features.dropna(subset=features)

        if latest_with_features.empty:
            print("Not enough valid feature data to generate a latest market prediction.")
        else:
            latest_features = latest_with_features.iloc[-1:][features]
            latest_predictions = {}  # Prepare a dictionary for latest predictions
            for model_name, model in trained_models.items():
                if model_name in ['RandomForest', 'XGBoost']:
                    prediction = get_clear_prediction(model, latest_features)
                    print(f"{model_name} predicts tomorrow's market will move: {prediction}")
                    latest_predictions[model_name] = prediction

        # === Final Strategy Value ===
        print("\n=== Final Strategy Values ===")
        value_cols = [col for col in returns_df.columns if col.endswith('_Value')]
        for col in value_cols:
            model = col.replace('_Value', '')
            final_value = returns_df[col].dropna().iloc[-1]
            print(f"{model}: ${final_value:.2f}")

        # Plot strategy performance
        plot_results(returns_df)

        # Feature importance
        print("\n=== Feature Importance Analysis ===")
        for model, importances in feature_importances.items():
            print(f"\n{model} Top Features:")
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:5]:
                print(f"{feature}: {importance:.4f}")

        # === Display Dashboard Summary ===
        print("\n=== Displaying Dashboard Summary ===")
        display_dashboard_summary(returns_df, regime_analysis, wf_metrics, latest_predictions)



# In[147]:


if __name__ == "__main__":
    main()


# In[ ]:




