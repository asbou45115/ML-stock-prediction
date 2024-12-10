import pandas as pd
import numpy as np
import os
import sys
import subprocess
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

if len(sys.argv) != 2:
    print("Usage: python predictor.py STOCK_SYMBOL")
    sys.exit(1)

stock = sys.argv[1].upper().strip()

def load_filename():
    filename = f'data/{stock}_stock_data.csv'

    # If file does exist use data otherwise generate data with data_gen.py
    if not os.path.exists(filename):
        print(f'File {filename} NOT FOUND. Generating data...')
        try:
            subprocess.run(['python', 'data/data_gen.py', stock], check=True)
            filename = f'{stock}_stock_data.csv'
        except subprocess.CalledProcessError as e:
            print(f"Error generating data for {stock}: {e}")
            sys.exit(1)
            
    return filename

"""
Load stock data from a CSV file and preprocess the DataFrame.

Args:
    filename (str): Path to the CSV file containing stock data.

Returns:
    pandas.DataFrame: Preprocessed DataFrame with stock data, sorted from oldest to newest.
"""
def load_data(filename):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(filename, parse_dates=['timestamp'], index_col='timestamp')
    
    # Ensure all numeric fields are correctly formatted (in case of commas or strings)
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
    
    # Reverse the DataFrame to have oldest dates first
    df = df.iloc[::-1]

    return df

"""
Preprocess stock price data by extracting close prices and normalizing.

Args:
    df (pandas.DataFrame): Input DataFrame with stock data.

Returns:
    tuple: A tuple containing:
        - numpy.ndarray: Scaled close price data
        - MinMaxScaler: Scaler used for normalization
        - pandas.Index: Original dates of the DataFrame
"""
def preprocess_data(df):
    # Select the 'close' price and reshape it
    close_data = df['close'].values.reshape(-1, 1)
    
    # Normalize data using MinMaxScaler to scale values between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)

    return scaled_data, scaler, df.index

"""
Create input sequences and corresponding target values for LSTM model.

Args:
    data (numpy.ndarray): Scaled stock price data.
    time_step (int, optional): Number of previous time steps to use for prediction. 
                                Defaults to 50.

Returns:
    tuple: A tuple containing:
        - numpy.ndarray: Input sequences (X)
        - numpy.ndarray: Target values (y)
"""
def create_sequences(data, time_step=50):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step, 0])
        y.append(data[i+time_step, 0])
    return np.array(X), np.array(y)

"""
Build an LSTM neural network model for stock price prediction.

Args:
    input_shape (tuple): Shape of the input data (time steps, features).

Returns:
    tensorflow.keras.models.Sequential: Compiled LSTM neural network model.
"""
def build_lstm_model(input_shape):
    model = Sequential()
    
    # Adding the first LSTM layer with Dropout
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    # Adding a second LSTM layer with Dropout
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Adding Dense layers for output
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

"""
Train the LSTM model and generate predictions.

Args:
    model (tensorflow.keras.models.Sequential): LSTM neural network model.
    X_train (numpy.ndarray): Training input sequences.
    y_train (numpy.ndarray): Training target values.
    X_test (numpy.ndarray): Testing input sequences.
    y_test (numpy.ndarray): Testing target values.
    scaler (MinMaxScaler): Scaler used for normalization.

Returns:
    tuple: A tuple containing:
        - numpy.ndarray: Rescaled predicted values
        - numpy.ndarray: Rescaled actual test values
        - tensorflow.keras.models.Sequential: Trained model
"""
def train_evaluate_model(model, X_train, y_train, X_test, y_test, scaler):
    # Train the model
    model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.1)
    
    # Make predictions on test data
    predictions = model.predict(X_test)
    
    # Inverse transform the predicted values back to original scale
    predictions_rescaled = scaler.inverse_transform(predictions)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    return predictions_rescaled, y_test_rescaled, model

"""
Create a plot of actual vs predicted stock prices and future predictions.

Args:
    y_test (numpy.ndarray): Actual test stock prices.
    y_pred (numpy.ndarray): Predicted stock prices for test data.
    test_dates (pandas.Index): Dates corresponding to test data.
    future_predictions (numpy.ndarray, optional): Predicted future stock prices. Defaults to None.
    future_dates (pandas.DatetimeIndex, optional): Dates for future predictions. Defaults to None.
"""
def plot_predictions(y_test, y_pred, test_dates, future_predictions=None, future_dates=None):
    plt.figure(figsize=(16, 8))
    
    # Plot test data (actual vs predicted)
    plt.plot(test_dates, y_test, color='blue', label='Actual Stock Price')
    plt.plot(test_dates, y_pred, color='red', label='Predicted Stock Price')
    
    # Plot future predictions if available
    if future_predictions is not None and future_dates is not None:
        plt.plot(future_dates, future_predictions, color='green', label='Future Price Predictions')
    
    plt.title(f'{stock} Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    plt.legend()
    
    # Rotate and align the tick labels so they look better
    plt.gcf().autofmt_xdate()
    
    # Use a more readable date format
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.tight_layout()
    plt.savefig(f'{stock}_stock_prediction.png')
    plt.close()

"""
Predict future stock prices using the trained LSTM model.

Args:
    model (tensorflow.keras.models.Sequential): Trained LSTM model.
    last_sequence (numpy.ndarray): Last sequence of historical prices used for prediction.
    scaler (MinMaxScaler): Scaler used for normalization.
    test_dates (pandas.Index): Dates corresponding to the test data.
    days_to_predict (int, optional): Number of future days to predict. Defaults to 30.

Returns:
    tuple: A tuple containing:
        - numpy.ndarray: Predicted future stock prices
        - pandas.DatetimeIndex: Dates for future predictions
"""
def predict_future_prices(model, last_sequence, scaler, test_dates, days_to_predict=30):
    current_sequence = last_sequence.copy()
    future_predictions = []
    
    # Create future dates
    last_date = test_dates[-1]
    future_dates = pd.date_range(start=last_date, periods=days_to_predict+1)[1:]

    for _ in range(days_to_predict):
        # Reshape the current sequence for prediction
        current_input = current_sequence.reshape(1, current_sequence.shape[0], 1)
        
        # Predict the next price
        next_pred = model.predict(current_input)
        
        # Rescale the prediction
        next_pred_rescaled = scaler.inverse_transform(next_pred)[0][0]
        future_predictions.append(next_pred_rescaled)
        
        # Update the sequence by removing the oldest price and adding the new prediction
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred[0][0]

    return np.array(future_predictions), future_dates

def main():
    filename = load_filename()
    df = load_data(filename)
    
    scaled_data, scaler, original_dates = preprocess_data(df)
    X, y = create_sequences(scaled_data, time_step=50)
    
    # Split data into training and testing sets (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    model = build_lstm_model(input_shape=(X_train.shape[1], 1))
    y_pred_rescaled, y_test_rescaled, trained_model = train_evaluate_model(model, X_train, y_train, X_test, y_test, scaler)
    
    # Get dates for test data
    test_dates = original_dates[len(original_dates) - len(y_test_rescaled):]
    
    # Predict future prices
    last_sequence = scaled_data[-50:]  # Use the last 50 days of scaled prices
    future_predictions_scaled, future_dates = predict_future_prices(trained_model, last_sequence, scaler, test_dates, days_to_predict=30)
    
    # Plot historical and future predictions
    plot_predictions(y_test_rescaled, y_pred_rescaled, test_dates, future_predictions_scaled, future_dates)
    
    print("Predicted Future Stock Prices:")
    for date, price in zip(future_dates, future_predictions_scaled):
        print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")

if __name__ == '__main__':
    main()