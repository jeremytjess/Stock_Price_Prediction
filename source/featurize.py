#!/usr/bin/env python3
# Author: Jeremy Jess

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import os
import sys
import config

import pandas as pd
import numpy as np

from datetime import datetime as dt
import pandas_datareader as web


# featurizer class for transforming raw data into feature matrix
class Featurizer(object):

    def __init__(self,tickers):
        self.tickers = tickers
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
        for tic in self.tickers:
            tic_df = web.DataReader(tic,'yahoo',start,end)\
                [['Open','High','Low','Adj Close','Volume']].reset_index()
            tic_df.columns = [col.lower() for col in tic_df.columns]
            #tic_df['ticker'] = tic
            tic_df.set_index('date',inplace=True)
            df = pd.concat([df,tic_df],axis=0)

        # use adjusted close for simplicity
        df['close'] = df['adj close']
        del df['adj close']

        self.raw_df = df

    def featurize_historical_data(self,start=dt(2000,1,1),end=dt.now(),forward_lag=5):
        """ Transform historical stock price data into
            feature matrix

            Parameters
            __________
            forward_lag: int
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
        for lag in [1,2,3,4,5,6,7,8,9,10]:
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

    def split_train_test(self,train_percent=0.7):
        """ splits data up into train/test w/ labels

            Parameters
            ----------
            train_percent: 0 < float < 1
                percentage of data to be used for training

            Returns
            --------
            X_train: DataFrame
                training data
            X_test: DataFrame
                test data
            y_train: Series
                training labels
            y_test: Series
                test labels
        """
        train_size = int(self.X.shape[0]*train_percent)

        X_train = self.X[0:train_size]
        y_train = self.y[0:train_size]

        X_test = self.X[train_size:]
        y_test = self.y[train_size:]


        return X_train,X_test,y_train,y_test


#######################################################
# main
#######################################################

def main(tickers):

    # display relevant information
    print("Tickers = ",tickers)

    print()

    # get historical data
    print("Generating Historical Data ... ")
    data = Featurizer(tickers)
    data.process_historical_data()

    print()

    # featurize data
    print("Adding Features ...")
    data.featurize_historical_data(forward_lag=1)

    # data info
    n,d = data.X.shape
    print(f"Number of examples: {n}")
    print(f"Number of feautures: {d}")

    print()

    # split data
    print("Splitting data into training/test ... ")
    X_train,X_test,y_train,y_test = data.split_train_test(train_percent=0.6)

    print()

    # display training data dimensions
    n,d = X_train.shape
    print("Number of training samples: ",n)
    print("Number of training features: ",d)

    print()

    # display test data dimensions
    n,d = X_test.shape
    print("Number of test samples: ",n)
    print("Number of test features: ",d)

    print()

    # prefix for file names
    PREFIX = config.gen_prefix(tickers)

    print("Writing to file ... ")

    # historical data
    HISTORICAL_DATA_FILENAME = '../data/raw/'+PREFIX+'hist.csv'
    data.raw_df.to_csv(HISTORICAL_DATA_FILENAME)
    print(f'\{HISTORICAL_DATA_FILENAME}')

    # training features
    FEATURES_TRAIN_FILENAME = '../data/processed/'+PREFIX+'train_features.csv'
    X_train.to_csv(FEATURES_TRAIN_FILENAME)
    print(f'\{FEATURES_TRAIN_FILENAME}')

    # test features
    FEATURES_TEST_FILENAME = '../data/processed/'+PREFIX+'test_features.csv'
    X_test.to_csv(FEATURES_TEST_FILENAME)
    print(f'\{FEATURES_TEST_FILENAME}')

    # training labels
    LABELS_TRAIN_FILENAME = '../data/processed/'+PREFIX+'train_labels.csv'
    df_train_labels = pd.DataFrame(y_train)
    df_train_labels.columns = ['target']
    df_train_labels.to_csv(LABELS_TRAIN_FILENAME)
    print(f'\{LABELS_TRAIN_FILENAME}')

    # test labels
    LABELS_TEST_FILENAME = '../data/processed/'+PREFIX+'test_labels.csv'
    df_test_labels = pd.DataFrame(y_test)
    df_test_labels.columns = ['target']
    df_test_labels.to_csv(LABELS_TEST_FILENAME)
    print(f'\{LABELS_TEST_FILENAME}')


if __name__ == '__main__':
    main(sys.argv[1:])

