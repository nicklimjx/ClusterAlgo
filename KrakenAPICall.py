# Return dataframe of token prices

# Import dependencies
import requests
import pandas as pd

def get_series(pairs: list, timeframe: str, extract: list) -> pd.DataFrame:
    assert(len(pairs) == len(extract))

    table = pd.DataFrame()
    for index, token in enumerate(pairs):
        url = f'https://api.kraken.com/0/public/OHLC?pair={token}&interval={timeframe}'

        # Make the API request
        response = requests.get(url)
        data = response.json()
        
        # Extract OHLCV data
        ohlc_data = data['result'][extract[index]]

        # Convert the data to a DataFrame
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'wavg price', 'count']
        df = pd.DataFrame(ohlc_data, columns=columns)

        # Convert 'open', 'high', 'low', 'close', 'volume', 'wavg_price' to numeric type
        string_columns = ['open', 'high', 'low', 'close', 'volume', 'wavg price']
        df[string_columns] = df[string_columns].apply(pd.to_numeric, errors='coerce')
        df.set_index('timestamp', inplace = True)

        # Update table
        table[token] = df["close"]
    
    return table
