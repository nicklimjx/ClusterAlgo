import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

# my edited api req to account for kraken's weird api naming convention

def kraken_api_data(pair, interval, since):
    # interval is the timeframe interval in minutes
    # 1 5 15 30 60 240 1440 10080 21600

    # Fetch OHLCV data from Kraken

    # Make the API request
    response = requests.get(f'https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}&since={since}')
    data = response.json()

    # if statement for special case of BTC and ETH pairs
    if pair == 'BTCUSD':
        extract_index = 'XXBTZUSD'
    elif pair == 'ETHUSD':
        extract_index = 'XETHZUSD'
    elif pair == 'XRPUSD':
        extract_index = 'XXRPZUSD'
    else:
        extract_index = pair
        
    if data['error'] != []:
        return []

    ohlc_data = data['result'][extract_index]
    
    # Convert the data to a DataFrame
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'wavg price', 'count']
    df = pd.DataFrame(ohlc_data, columns=columns)

    # Convert Unix timestamp to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # Convert 'open', 'high', 'low', 'close', 'volume', 'wavg_price' to numeric type
    string_columns = ['open', 'high', 'low', 'close', 'volume', 'wavg price']
    df[string_columns] = df[string_columns].apply(pd.to_numeric, errors='coerce')
    df.set_index('timestamp', inplace = True)

    return df


# def get_market_cap(basket):
#     get market cap of basket and normalise

def normalise_min_max(df):
    return df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), axis=0)

def normalise_market_cap(basket):
    for asset in basket:
        asset = asset.apply(lambda x: x * get_market_cap(basket), axis=0)


# if all assets available then use this
# dex = ['UNI', 'RUNE', 'INJ', 'SNX', 'CAKE', 'CRV', 'SEI', '1INCH', 'OSMO', 'ZRX']
# cex = ['BNB', 'OKB', 'LEO', 'CRO', 'FTT', 'KCS', 'BGB', 'HT', 'GT', 'MX']
# defi = ['AVAX', 'LINK', 'LDO', 'AAVE', 'MKR', 'LUNC', 'SNX', 'FTM', 'KAVA', 'XDC']
# top20 = ['BTC', 'ETH', 'BNB', 'XRP', 'SOL', 'ADA', 'DOGE', 'AVAX', 'TRX', 'LINK', 'DOT', 'MATIC', 'TON', 'SHIB', 'LTC', 'BCH', 'ATOM', 'XLM', 'OKB', 'LEO']
# derivatives = ['FTT', 'SNX', 'GMX', 'RNB', 'UMA', 'MNGO', 'STPT', 'GNS', 'VEGA', 'PERP']

# test baskets
new_dex = ['UNI', 'RUNE', 'INJ', 'SNX', 'CRV', 'SEI', '1INCH', 'ZRX', 'KAVA']
new_defi = ['AVAX', 'LINK', 'LDO', 'AAVE', 'MKR', 'SNX', 'FTM', 'KAVA']

interval = 1440
for i in range(len(new_dex)):
    new_dex[i] = kraken_api_data(f'{new_dex[i]}USD', interval, 1627776000)
    print(normalise_min_max(new_dex[i]))