from flask import Flask, request, jsonify, render_template
import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close']
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler, scaled_data
def build_cnn_model(input_shape):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    return model
def train_svm(cnn_features, labels):
    svm = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svm.fit(cnn_features, labels)
    return svm
def predict_stock(cnn_model, svm, X_test, y_test, scaler, scaled_data, future_days, end_date):
    cnn_features_test = cnn_model.predict(X_test)
    svm_predictions = svm.predict(cnn_features_test)
    svm_predictions = scaler.inverse_transform(svm_predictions.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    future_predictions = []
    last_60_days = scaled_data[-60:]
    future_X = np.array([last_60_days[:, 0]])
    future_X = np.reshape(future_X, (future_X.shape[0], future_X.shape[1], 1))

    future_dates = [datetime.datetime.strptime(end_date, '%Y-%m-%d') + datetime.timedelta(days=i) for i in range(1, future_days + 1)]
    
    for _ in range(future_days):
        cnn_feature = cnn_model.predict(future_X)
        next_price_scaled = svm.predict(cnn_feature)
        next_price = scaler.inverse_transform(next_price_scaled.reshape(-1, 1))[0][0]
        future_predictions.append(next_price)

        new_input = np.append(future_X[0, 1:, 0], next_price_scaled)
        future_X = np.array([new_input])
        future_X = np.reshape(future_X, (future_X.shape[0], future_X.shape[1], 1))

    return future_dates, future_predictions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    tickers = data['ticker'].split(',')
    start_date = data['start_date']
    end_date = data['end_date']
    future_days = int(data['future_days'])
    
    stock_predictions = {}
    try:
        for ticker in tickers:
            stock_data = fetch_stock_data(ticker.strip(), start_date, end_date)
            X, y, scaler, scaled_data = preprocess_data(stock_data.values)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            cnn_model = build_cnn_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            cnn_model.compile(optimizer='adam', loss='mse')
            cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

            cnn_features_train = cnn_model.predict(X_train)
            svm = train_svm(cnn_features_train, y_train)

            future_dates, future_prices = predict_stock(cnn_model, svm, X_test, y_test, scaler, scaled_data, future_days, end_date)
            stock_predictions[ticker.strip()] = {"dates": [date.strftime('%Y-%m-%d') for date in future_dates], "prices": future_prices}
        return jsonify(stock_predictions)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
