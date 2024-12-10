import requests
import sys
from api_key import key

if len(sys.argv) != 2:
    print("Usage: python data_gen.py STOCK_SYMBOL")
    sys.exit(1)

stock_symbol = sys.argv[1].upper().strip()

url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock_symbol}&outputsize=full&apikey={key}&datatype=csv'

response = requests.get(url)

if response.status_code == 200:
    filename = f'{stock_symbol}_stock_data.csv'
    with open(filename, 'wb') as file:
        file.write(response.content)
    
    print(f'Downloaded stock data for {stock_symbol} and saved as {filename}')
else:
    print(f'Failed to retrieve data: {response.status_code}')
    