import pandas as pd
import numpy as np
import yfinance as yf

def store_stock_data(stock_name = 'TSLA'):
    """Function to retrieve stock data from yahoo."""
    stonk = yf.Ticker(stock_name) # gets stock data from yahoo
    hist = stonk.history(period="max") # historical stock prices
    hist.reset_index(inplace=True) # takes the date stamp out of the index column
    hist.rename(columns = {'Date':"DateTime"},inplace=True) # Changes the name of the date column
    hist['DateTime'] = pd.to_datetime(hist['DateTime'],utc=True) # Changes the timestamps to UTC
    hist.to_csv('../data/raw/'+stock_name+'_stock_price.csv')
    return

if __name__ == '__main__':
    main()