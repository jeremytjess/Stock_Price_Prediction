# Stock Prediction
## Overview
This program uses a variety of supervised machine learning spproaches, including Regression and Neural Nets to generate predictive models for stock prices
from historical data supplied by the Yahoo Finance API.  

## Installation 
To install required libraries, run ```pip install -r requirements.txt```

### featurize.py
Collects historical data from a specified time period into a Pandas DataFrame, adds additional time series features, splits data into 
train/test sets, and saves the results into the *data/* directory. Additional features include:
  * Simple Moving Average (5,10,20,50,100,200 days)
  * Bollinger Bands (10/20 days & 1/2 stdev)
  * Donchian Channels (5,10,20,50,100,200 days)
  * Shifts (1-10 days)

**To run**:
```python featurize.py <TICKER>``` where <TICKER> is the stock ticker you want data for.  

**Output**:
  * data/raw/{TICKER}_hist.csv (raw data collected from Yahoo Finance)
  * data/processed/{TICKER}_train_features.csv (training data features)
  * data/processed/{TICKER}_test_features.csv (test data examples)
  * data/processed/{TICKER}_train_labels.csv (training data labels)
  * data/processed/{TICKER}_test_labels.csv (test data labels)


### predictor.py
Evaluates data using models specified in *models.py* on data for provided ticker and outputs the results in Plotly plots.

**To run**:
```python predictor.py <TICKER>``` 

**Output**:
  * metrics
    * Mean Squared Error
    * Mean Absolute Error
    * R1 Score
  * plots
    * subplot of training/test/predicted values for each model
  
### models.py
Contains all models to be used to evaluate dataset.  

Current models being used:
  * Dummy (for baseline)
  * Linear Regression
  * Linear Regression w/ Bagging
  * Linear Regression w/ Boosting
  * K-Nearest Neighbors (likely will deprecate soon)
  * Gradient Boosting
  * MLP (Neural Net)
  
### preprocessors.py
Contains the preprocessing steps for pipeline.  

**Current Pre-Processing Steps**:
  * MinMaxScaler(-1,1)
  
### config.py
Contains various helper functions.  
  * ```gen_prefix(tickers)```:  
    * generates prefix based on ticker for csv filenames
  * ```reduce_features(X_train,X_test,y_train,count)```:
    * reduces the number of features in training and test datset to *count* using Pearson's Correlation
  * ```scrape_dow_tickers()```:
    * scrapes Dow Jones tickers from Wikipedia and populates them into a list
  * ```scrape_sp_tickers()```:
    * scapes S&P 500 tickers from Wikipedia and populates them into a list

  
  
  
