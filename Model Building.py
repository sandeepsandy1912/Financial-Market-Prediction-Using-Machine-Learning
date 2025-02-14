import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from keras_tuner import RandomSearch
from scipy.stats import randint
import pickle
import time

# Download stock data
stocks = ['AAPL', 'MSFT']
stock_data_1 = {ticker: yf.download(ticker, start="2018-01-01", end="2023-12-31") for ticker in stocks}

# Download forex data
forex_pairs = ['EURUSD=X', 'JPY=X']
forex_data_1 = {ticker: yf.download(ticker, start="2018-01-01", end="2023-12-31") for ticker in forex_pairs}

# Download crypto data
cryptos = ["BTC-USD", "ETH-USD"]
crypto_data_1 = {ticker: yf.download(ticker, start="2020-01-01", end="2023-12-31") for ticker in cryptos}

def preprocess_data(data):
    for ticker, df in data.items():
        df.fillna(method='ffill', inplace=True)
        df.dropna(inplace=True)
    return data

# Preprocessed data
pre_stock_data = preprocess_data(stock_data_1)
pre_forex_data = preprocess_data(forex_data_1)
pre_crypto_data = preprocess_data(crypto_data_1)

def prepare_features(data):
    features = {}
    for ticker, df in data.items():
        df['Return'] = df['Close'].pct_change()
        df.dropna(inplace=True)
        scaler = StandardScaler()
        df[['Close', 'Volume', 'Return']] = scaler.fit_transform(df[['Close', 'Volume', 'Return']])
        features[ticker] = df[['Close', 'Volume', 'Return']]
    return features

# Prepare features
stock_features = prepare_features(pre_stock_data)
forex_features = prepare_features(pre_forex_data)
crypto_features = prepare_features(pre_crypto_data)

def add_polynomial_features(df, degree=2):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    original_features = df[['Close','Return']]
    poly_features = poly.fit_transform(original_features)
    feature_names = poly.get_feature_names_out(original_features.columns)
    poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
    return poly_df

# Apply polynomial features to datasets
stock_features_poly = {ticker: add_polynomial_features(df) for ticker, df in stock_features.items()}
forex_features_poly = {ticker: add_polynomial_features(df) for ticker, df in forex_features.items()}
crypto_features_poly = {ticker: add_polynomial_features(df) for ticker, df in crypto_features.items()}

def prepare_data_for_modeling(df, scaler=None):
    X = df.drop(columns=['Close']).values
    y = df['Close'].values
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, y, scaler

def optimize_random_forest(X, y):
    rf = RandomForestRegressor(random_state=42)
    param_dist = {
        'n_estimators': randint(50, 100),  # Reduced range
        'max_depth': [None] + list(range(5, 10)),  # Reduced depth
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2'],  # Removed None for simplicity
    }
    rf_random = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=20, cv=3,
                                scoring='neg_mean_squared_error', random_state=42, n_jobs=-1, verbose=1)
    rf_random.fit(X, y)
    return rf_random.best_estimator_

def optimize_gradient_boosting(X, y):
    gb = GradientBoostingRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],  # Reduced number of estimators
        'learning_rate': [0.01, 0.05],  # Common learning rates
        'max_depth': [3, 5],
        'min_samples_split': (2, 4),
        'min_samples_leaf': (1, 2),
    }
    gb_random = GridSearchCV(gb, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error',
                          n_jobs=-1, verbose=1)
    gb_random.fit(X, y)
    return gb_random.best_estimator_

def build_lstm_model(hp):
    model = Sequential()
    model.add(Input(shape=(6,1)))
    model.add(LSTM(units=hp.Int('units_1', min_value=32, max_value=128, step=32),
                   return_sequences=True))
    model.add(Dropout(hp.Float('dropout_1', 0.2, 0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units_2', min_value=32, max_value=128, step=32)))
    model.add(Dropout(hp.Float('dropout_2', 0.2, 0.5, step=0.1)))
    model.add(Dense(1))

    model.compile(optimizer=hp.Choice('optimizer', ['adam']),
                  loss='mse', metrics=['mae'])
    return model

def optimize_lstm(X_train, y_train):
    tuner = RandomSearch(
        build_lstm_model,
        objective='val_loss',
        max_trials=5,  # Reduced trials
        executions_per_trial=1,
        directory='lstm_tuning',
        project_name='lstm_opt',
        overwrite=True
    )

    tuner.search(
        np.expand_dims(X_train, axis=-1), y_train,
        epochs=5,  
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model

def optimize_svr(X, y):
    svr = SVR()
    param_dist = {
        'C': np.logspace(-2, 3, 5),
        'kernel': ['linear', 'rbf'],
        'epsilon': np.linspace(0.01, 0.5, 5),
        'gamma': ['scale']
    }
    svr_random = RandomizedSearchCV(
        svr, 
        param_distributions=param_dist, 
        n_iter=20,
        cv=3,
        scoring='neg_mean_squared_error', 
        random_state=42, 
        n_jobs=-1, 
        verbose=0
    )
    svr_random.fit(X, y)
    return svr_random.best_estimator_

def optimize_ridge(X, y):
    ridge = Ridge()
    param_grid = {'alpha': np.logspace(-4, 4, 20)}
    ridge_grid = GridSearchCV(
        ridge, 
        param_grid=param_grid, 
        cv=3, 
        scoring='neg_mean_squared_error', 
        n_jobs=-1, 
        verbose=0
    )
    ridge_grid.fit(X, y)
    return ridge_grid.best_estimator_

# Initialize dictionaries for storing best models
best_rf_models = {'Stocks': {}, 'Forex': {}, 'Cryptocurrencies': {}}
best_gb_models = {'Stocks': {}, 'Forex': {}, 'Cryptocurrencies': {}}
enhanced_lstm_models = {'Stocks': {}, 'Forex': {}, 'Cryptocurrencies': {}}
best_svr_models = {'Stocks': {}, 'Forex': {}, 'Cryptocurrencies': {}}
best_ridge_models = {'Stocks': {}, 'Forex': {}, 'Cryptocurrencies': {}}

# Train models for each financial market
def train_and_store_models(features_poly, model_name, model_function):
    best_models = {'Stocks': {}, 'Forex': {}, 'Cryptocurrencies': {}}
    for features, name in zip(features_poly, ["Stocks", "Forex", "Cryptocurrencies"]):
        for ticker, df in features.items():
            try:
                # Prepare data
                X, y, _ = prepare_data_for_modeling(df)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train the model
                start_time = time.time()
                best_models[name][ticker] = model_function(X_train, y_train)
                elapsed_time = time.time() - start_time

                print(f"{model_name} training time for {name} - {ticker}: {elapsed_time:.4f} seconds")

            except Exception as e:
                print(f"Error training {model_name} for {ticker}: {e}")

    return best_models

# Train all models
best_rf_models = train_and_store_models([stock_features_poly, forex_features_poly, crypto_features_poly],
                                        "Random Forest", optimize_random_forest)
best_gb_models = train_and_store_models([stock_features_poly, forex_features_poly, crypto_features_poly],
                                        "Gradient Boosting", optimize_gradient_boosting)
enhanced_lstm_models = train_and_store_models([stock_features_poly, forex_features_poly, crypto_features_poly],
                                              "LSTM", optimize_lstm)
best_svr_models = train_and_store_models([stock_features_poly, forex_features_poly, crypto_features_poly],
                                         "SVR", optimize_svr)
best_ridge_models = train_and_store_models([stock_features_poly, forex_features_poly, crypto_features_poly],
                                           "Ridge Regressor", optimize_ridge)

# Function to evaluate and get metrics for each model
def evaluate_and_get_metrics(models, X_test, y_test):
    results = {}
    for model_name, model in models.items():
        if model_name == "LSTM":
            X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1],1 ))
            y_pred = model.predict(X_test_reshaped).flatten()
        else:
            y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Calculate directional accuracy
        y_test_direction = np.sign(np.diff(y_test))
        y_pred_direction = np.sign(np.diff(y_pred))
        directional_accuracy = np.mean(y_test_direction == y_pred_direction)

        # Calculate accuracy
        y_test_binary = (y_test > 0).astype(int)
        y_pred_binary = (y_pred > 0).astype(int)
        accuracy = accuracy_score(y_test_binary, y_pred_binary)

        results[model_name] = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'Directional Accuracy': directional_accuracy,
            'Accuracy': accuracy,
            'Predictions': y_pred,
            'Actual': y_test
        }
    return results

# Function to evaluate models for each market
def evaluate_financial_market(features, ridge_models, svr_models, rf_models, gb_models, enhanced_lstm_models):
    models_dict = {
        "Ridge Regression": ridge_models,
        "SVR": svr_models,
        "Random Forest": rf_models,
        "Gradient Boosting": gb_models,
        "LSTM": enhanced_lstm_models
    }

    evaluation_results = {}

    for ticker, df in features.items():
        # Prepare data
        X, y, scaler = prepare_data_for_modeling(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Prepare a dictionary of models for the current ticker
        models = {model_name: models_dict[model_name][ticker] for model_name in models_dict}

        # Evaluate models
        metrics = evaluate_and_get_metrics(models, X_test, y_test)
        evaluation_results[ticker] = metrics

    # Convert nested dictionary to DataFrame
    evaluation_df = pd.DataFrame({(ticker, metric): values[metric] for ticker, values in evaluation_results.items() for metric in values}).T

    # Properly format the MultiIndex DataFrame
    evaluation_df.index.names = ['Ticker', 'Model']

    return evaluation_df

# Save all evaluation results to a single file
def save_consolidated_evaluation_results():
    # Consolidate all evaluation results
    consolidated_results = {}
    for features, name in zip(
        [stock_features_poly, forex_features_poly, crypto_features_poly],
        ["Stocks", "Forex", "Cryptocurrencies"],
    ):
        ridge_models = best_ridge_models[name]
        svr_models = best_svr_models[name]
        rf_models = best_rf_models[name]
        gb_models = best_gb_models[name]
        lstm_models = enhanced_lstm_models[name]

        evaluation_results = evaluate_financial_market(
            features, ridge_models, svr_models, rf_models, gb_models, lstm_models
        )
        
        # Store in consolidated dictionary
        consolidated_results[name] = evaluation_results

    # Save the consolidated evaluation results
    with open("consolidated_evaluation_results.pkl", "wb") as file:
        pickle.dump(consolidated_results, file)

# Select and save the best models based on MSE
def select_and_save_best_models():
    best_models_summary = {}

    # Load the consolidated evaluation results
    with open("consolidated_evaluation_results.pkl", "rb") as file:
        consolidated_results = pickle.load(file)

    for market, evaluation_results in consolidated_results.items():
        best_models_summary[market] = {}

        for symbol in evaluation_results.index.levels[0]:
            # Get the metrics for all models for the current symbol
            metrics_df = evaluation_results.loc[symbol]

            # Convert 'MSE' column to numeric, forcing errors to NaN
            metrics_df['MSE'] = pd.to_numeric(metrics_df['MSE'], errors='coerce')

            # Drop rows with NaN MSE values
            metrics_df = metrics_df.dropna(subset=['MSE'])

            # Select the best two models based on MSE
            best_models = metrics_df.nsmallest(2, 'MSE')

            best_models_summary[market][symbol] = best_models

    # Save the best models summary
    with open("best_models_summary.pkl", "wb") as file:
        pickle.dump(best_models_summary, file)

# Run the functions to save the results
save_consolidated_evaluation_results()
select_and_save_best_models()
