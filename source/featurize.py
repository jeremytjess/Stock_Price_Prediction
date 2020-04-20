#!/usr/bin/env python3
# Author: Jeremy Jess

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import sys
import config

import pandas as pd

from datetime import datetime as dt
import pandas_datareader as web


# featurizer class for transforming raw data into feature matrix
class Featurizer(object):

    def __init__(self,ticker):
        self.ticker = ticker
        self.raw_df = None
        self.feature_df = None
        self.X = None
        self.y = None

    def process_historical_data(self,start=dt(2000,1,1),end=dt.now()):
        """ generate historical data

            Parameters
            __________
            start: datetime
                beginning of historical data
            end: datetime
                end of historical data

            Generates
            ---------
            raw_df: DataFrame
               dataframe with raw historical data
        """
        df = pd.DataFrame()

    	# gather raw data
        df = web.DataReader(self.ticker,'yahoo',start,end)\
            [['Open','High','Low','Adj Close','Volume']].reset_index()
        df.columns = [col.lower() for col in df.columns]
        df.set_index('date',inplace=True)

        # use adjusted close for simplicity
        df['close'] = df['adj close']
        del df['adj close']

        self.raw_df = df

    def featurize_historical_data(self,start=dt(2000,1,1),end=dt.now(),forward_lag=5):
        """ Transform historical stock price data into
            feature matrix

            Parameters __________ forward_lag: int
                amount of days to generate prediction

            Returns
            ---------
            feature_df: DataFrame
               dataframe with all additional features
        """
        df = self.raw_df.copy()

        # calculate simple moving averages
        for sma_period in [5,10,20,50,100,200]:
            indicator_name = "sma_%d" % (sma_period)
            df[indicator_name] = df['close'].rolling(sma_period).mean()

        # caculate various bollinger bands
        # bollinger bands = mean(period) +/- n*stdev(period)
        df['bollingerband_up_20_2'] = df['close'].rolling(20).mean() + 2*df['close'].rolling(20).std()
        df['bollingerband_down_20_2'] = df['close'].rolling(20).mean() - 2*df['close'].rolling(20).std()
        df['bollingerband_up_20_1'] = df['close'].rolling(20).mean() + df['close'].rolling(20).std()
        df['bollingerband_down_20_1'] = df['close'].rolling(20).mean() - df['close'].rolling(20).std()
        df['bollingerband_up_10_1'] = df['close'].rolling(10).mean() + df['close'].rolling(10).std()
        df['bollingerband_down_10_1'] = df['close'].rolling(10).mean() - df['close'].rolling(10).std()
        df['bollingerband_up_10_2'] = df['close'].rolling(10).mean() + 2*df['close'].rolling(10).std()
        df['bollingerband_down_10_2'] = df['close'].rolling(10).mean() - 2*df['close'].rolling(10).std()

        # calculate donchian channels
        for channel_period in [5,10,20,50,100,200]:
            up_name = f'donchian_channel_up_{channel_period}'
            down_name = f'donchian_channel_down_{channel_period}'
            df[up_name] = df['high'].rolling(channel_period).max()
            df[down_name] = df['low'].rolling(channel_period).min()

        newdata = df['close'].to_frame()

        # shift data for specified time periods
        #for lag in [1,5,10,20,50,100]:
        for lag in range(1,11):
            shift = lag
            shifted = df.shift(shift)
            shifted.columns = \
                [str.format("%s_shifted_by_%d" % (column ,shift)) for column in shifted.columns]
            newdata = pd.concat((newdata,shifted),axis=1)

        # shift the forecast window
        newdata['target'] = newdata['close'].shift(-forward_lag)

        newdata = newdata.drop('close',axis=1)
        newdata = newdata.dropna()

        self.feature_df = newdata
        self.X = newdata.drop("target",axis=1)
        self.y = newdata['target']

        return self.feature_df



#######################################################
# main
#######################################################

def main(argv):

    if len(argv) < 3:
        print("Must be in format: python featurize.py  <TICKER> <FORWARD_LAG>")
        exit(0)
    elif not int(argv[2]):
        print("Must be in format: python featurize.py  <TICKER> <FORWARD_LAG>")
        exit(0)

    # relevant variables
    ticker = argv[1]
    forward_lag = int(argv[2])

    if len(argv) > 3:
        start_date = dt.strptime(argv[3],'%Y-%m-%d')
        end_date = dt.strptime(argv[4],'%Y-%m-%d')
    else:
        start_date = dt(2000,1,1)
        end_date = dt.now()

    # display relevant information
    print("Ticker: ",ticker)

    print()

    # get historical data
    print("Generating Historical Data ... ")
    data = Featurizer(ticker)
    data.process_historical_data()

    print()

    # relevant stats
    print(f"Start Date: {str(start_date)[:10]}")
    print(f"Date: {str(end_date)[:10]}")
    print(f"Total Number of Days: {data.raw_df.shape[0]}")
    print(f"Prediction window: {forward_lag} days")

    print()

    # featurize data
    print("Adding Features ...")
    data.featurize_historical_data(start=start_date,
                                   end=end_date,
                                   forward_lag=forward_lag)

    # data info
    n,d = data.X.shape
    print(f"Number of examples: {n}")
    print(f"Number of feautures: {d}")

    print()

    # prefix for file names
    PREFIX = config.gen_prefix(ticker,forward_lag)

    print("Writing to file ... ")

    # historical data
    RAW_PREFIX = PREFIX.replace(f'_{forward_lag}','')
    HISTORICAL_DATA_FILENAME = f'../data/raw/{RAW_PREFIX}_hist.csv'
    data.raw_df.to_csv(HISTORICAL_DATA_FILENAME)
    print(f'\{HISTORICAL_DATA_FILENAME}')

    X = data.X
    y = data.y

    # training features
    FEATURES_FILENAME = f'../data/processed/{PREFIX}_features.csv'
    X.to_csv(FEATURES_FILENAME)
    print(f'\{FEATURES_FILENAME}')

    # test features
    LABELS_FILENAME = f'../data/processed/{PREFIX}_labels.csv'
    df_labels = pd.DataFrame(y)
    df_labels.columns = ['target']
    y.to_csv(LABELS_FILENAME)
    print(f'\{LABELS_FILENAME}')


if __name__ == '__main__':
    main(sys.argv)

