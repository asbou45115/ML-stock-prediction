# Stock Price Prediction

This project uses machine learning techniques to predict stock prices. The data is retrieved from Alpha Vantage API, processed, and used to train an LSTM (Long Short-Term Memory) model for stock price prediction. The code is divided into several components, including data generation, preprocessing, model building, and prediction.

## Project Overview
The goal of this project is to predict future stock prices based on historical data using LSTM neural networks. The project is structured to:

Download stock data: Retrieve stock price data using the Alpha Vantage API.
Data Preprocessing: Normalize the stock data and prepare it for the LSTM model.
Model Building: Create and train an LSTM model for stock price prediction.
Prediction and Visualization: Predict future stock prices and visualize the results.

## Project Structure
```bash
.
├── data/
│   ├── data_gen.py                # Script to download stock data from Alpha Vantage API
│   ├── <stock_symbol>_stock_data.csv  # Stock data for a given stock symbol (auto-generated)
├── predictor.py                   # Main script to run stock prediction
├── api_key.py                     # Store your API key for Alpha Vantage
├── requirements.txt               # List of dependencies
└── README.md                      # Project README file (this file)
```

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
```

## 2. Install Dependencies
Make sure you have Python 3.10+ installed. Install the necessary dependencies using pip:

```bash
pip install -r requirements.txt
```

requirements.txt includes the following libraries:

requests
pandas
numpy
scikit-learn
tensorflow
matplotlib

## 3. Set Up API Key
Get your Alpha Vantage API key from Alpha Vantage.
Create a file named api_key.py in the project root directory, and define your API key in it:

### api_key.py
```bash
key = 'your_alpha_vantage_api_key'
```

# Usage

## 1. Run the Stock Data Generator
To generate the stock data for a specific symbol, run the following command. The data will be saved in the data/ folder.

```bash
python data/data_gen.py STOCK_SYMBOL
```

## 2. Run the Stock Price Prediction
To use the generated data for training the LSTM model and making predictions, run the following command:

```bash
python predictor.py STOCK_SYMBOL
```

## 3. Output
The script will output a prediction plot, showing actual vs predicted stock prices.

The plot will be saved as a PNG image with the filename <stock_symbol>_stock_prediction.png.


### Example Workflow
Download Stock Data: If the stock data for the given symbol does not exist, the script will invoke data_gen.py to generate the data (You can run the predictor without having the data created).
Train the Model: The stock data will be processed, and an LSTM model will be trained to predict future stock prices.
Predict Future Prices: The trained model will predict future stock prices and visualize the results.
