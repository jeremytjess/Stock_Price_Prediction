import re
import requests
import json

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from tqdm import tqdm

from datetime import datetime as dt

def scrape_sp_tickers():
    response = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = BeautifulSoup(response.text, 'lxml')
    sp_tickers = ['^GSPC']

    table = soup.find('table',{'id':'constituents'})

    for row in table.findAll('tr')[1:]:
        row_text = row.findAll('td')
        ticker = row_text[0].text.replace("\n", "")
        sp_tickers.append(ticker)

    return sp_tickers


def get_page_contents(ticker):
    """returns html from page"""
    url = f'https://finance.yahoo.com/quote/{ticker}/key-statistics?p={ticker}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text,'lxml')

    return soup

def convert_to_float(val):
    try:
        if val == 'N/A':
            val = np.nan
        elif val[-1] == 'M':
            val = round(float(val[:-1])*1e6,2)
        elif val[-1] == 'B':
            val = round(float(val[:-1])*1e6,2)
        elif val[-1] == '%':
            val = round(float(val[:-1])*1e-2,4)
        else:
            val = round(float(val),2)
    except:
        try:
            val = str(dt.strptime(val,'%b %d, %Y'))[:10]
        except:
            val = val

    return val

def get_financial_measures(ticker):
    """gets rest of financial data"""
    soup = get_page_contents(ticker)
    financial_measures = dict()
    tables = soup.find_all('table')
    for table in tables:
        rows = table.find_all('tr',{'class':re.compile('Bxz(bb)*')})
        for row in rows:
            key = row.find('span').text
            values = [val.text for val in row.find_all('td',{'class':re.compile('Ta(c)*')})]
            financial_measures[key] = convert_to_float(values[0])
    return financial_measures


goog_finance = get_financial_measures('GOOG')
fb_finance = get_financial_measures('FB')

sp_tickers = scrape_sp_tickers()

dct = {}
for ticker in tqdm(sp_tickers,desc='tickers'):
    try:
        finance = get_financial_measures(ticker)
        dct[ticker] = finance
    except:
       print(f"failed to add {ticker}")


df = pd.DataFrame.from_dict(dct,orient='index')
df.to_csv('company_financials.csv')


