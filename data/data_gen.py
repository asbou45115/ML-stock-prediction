import requests
from api_key import key

stock_symbol = input("Enter Stock Symbol: ").upper().strip()

url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock_symbol}&outputsize=full&apikey={key}&datatype=csv'

response = requests.get(url)

if response.status_code == 200:
    filename = f'{stock_symbol}_stock_data.csv'
    with open(filename, 'wb') as file:
        file.write(response.content)
    
    print(f'Downloaded stock data for {stock_symbol} and saved as {filename}')
else:
    print(f'Failed to retrieve data: {response.status_code}')
    