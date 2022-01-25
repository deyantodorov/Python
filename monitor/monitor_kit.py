import string
import pandas as pd
from pandas_datareader import data
from datetime import datetime


def load_data(name: string, ticker: string, end: datetime, years: int):
    '''
    Load data from Yahoo Finance by ticker, start and end date
    '''
    end_date = end
    # print(end_date)

    start_date = end_date.replace(year=end_date.year - years)
    # print(start_date)

    filename = f'{name}_{start_date.month}_{start_date.year}_{end_date.month}_{end_date.year}.pkl'

    # print(filename)

    try:
        result = pd.read_pickle(filename)
    except FileNotFoundError:
        result = data.DataReader(ticker, 'yahoo', start_date, end_date)
        result.to_pickle(filename)

    return result
