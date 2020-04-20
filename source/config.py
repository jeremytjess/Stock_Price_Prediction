import requests
import numpy as np
import pandas as pd
#from bs4 import BeautifulSoup

METRICS = ["accuracy", "auroc", "f1_score", "sensitivity", "specificity", "precision"]



# generates prefix for filenames
def gen_prefix(ticker,lag):
    prefix = f'{ticker}_{lag}'
    return prefix

def reduce_features(X_train,X_test,y_train,count=50):
        """ trim feature count to specified amount

            Parameters
            -----------
            count: int
                amount of features to preserve

            Returns
            ----------
            X_train: DataFrame
                training df with only important features
            X_test: DataFrame
                test df with only important features
        """
        #X_train,y_train = pd.DataFrame(X_train),pd.DataFrame(y_train)
        #print(y_train.index)
        correlations = np.abs(X_train.corrwith(y_train))
        features =  list(correlations.sort_values(ascending=False)[0:count].index)
        X_train = X_train[features]
        X_test = X_test[features]
        return X_train,X_test





# scrapes wikipedia for DJI tickers
def scrape_dow_tickers():
    url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    dow_tickers = ['^DJI']
    table = soup.find('table',{'id':'constituents'})
    for row in table.findAll('tr')[1:]:
        row_text = row.findAll('td')
        ticker = row_text[2].text.replace("NYSE:\xa0", "").replace("\n","")
        dow_tickers.append(ticker)
    return dow_tickers

#scrapes wikipedia for s&p tickers
def scrape_sp_tickers():
    url = 'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text,'lxml')
    sp_tickers = ['^GSCPC']
    table = soup.find('table',{'id':'constituents'})

    for row in table.findAll('tr')[1:]:
        row_text = row.findAll('td')
        ticker = row_text[0].text.replace("\n", "")
        sp_tickers.append(ticker)
    return sp_tickers

