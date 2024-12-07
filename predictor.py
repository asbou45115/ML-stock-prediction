import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# stock = 'QQQ'
stock = 'SPY'
filename = f'data/{stock}_stock_data.csv'

def load_data(filename):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(filename, parse_dates=['timestamp'], index_col='timestamp')
    
    # Ensure all numeric fields are correctly formatted (in case of commas or strings)
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)

    return df

# Responsible for preprocessing data for the LSTM model
#
# Params: 
#       df -> pandas DataFrame 
def preprocess_data(df):
    # Select the 'close' price and reshape it
    close_data = df['close'].values.reshape(-1, 1)
    
    # Normalize data using MinMaxScaler to scale values between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)

    return scaled_data, scaler

# Creates the sequences for the LSTM model
#
# Params: 
#       data -> scaled_data from preprocess
def create_sequences(data, time_step=50):
    X, y = [], []
    for i in range(time_step, len(data)):
        # CSV file ordered in descending order of timestamp so we prepend to ensure trained data is from older set
        X.insert(0, data[i-time_step:i, 0])
        y.insert(0, data[i, 0])
    return np.array(X), np.array(y)

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

def train_evaluate_model(model, X_train, y_train, X_test, y_test, scaler):
    # Train the model
    model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.1)
    
    # Make predictions on test data
    predictions = model.predict(X_test)
    
    # Inverse transform the predicted values back to original scale
    predictions_rescaled = scaler.inverse_transform(predictions)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    return predictions_rescaled, y_test_rescaled

def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(14, 7))
    plt.plot(y_test, color='blue', label='Actual Stock Price')
    plt.plot(y_pred, color='red', label='Predicted Stock Price')
    plt.title(f'{stock} Price Prediction')
    plt.xlabel('Time (Days)')
    plt.ylabel('Stock Price (USD)')
    plt.legend()
    plt.savefig(f'{stock} stock_prediction.png')

# Main function to run the pipeline
def main():
    df = load_data(filename)
    scaled_data, scaler = preprocess_data(df)
    X, y = create_sequences(scaled_data, time_step=50)
    
    # Split data into training and testing sets (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    model = build_lstm_model(input_shape=(X_train.shape[1], 1))
    y_pred_rescaled, y_test_rescaled = train_evaluate_model(model, X_train, y_train, X_test, y_test, scaler)
    
    plot_predictions(y_test_rescaled, y_pred_rescaled)

if __name__ == '__main__':
    main()
