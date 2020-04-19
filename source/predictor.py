#!/usr/bin/env python3
# Author: Jeremy Jess

import pandas as pd
import numpy as np
import warnings
import joblib
import argparse
import os
import sys
import config
import classifiers
import preprocessors

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,TimeSeriesSplit,KFold,cross_val_score
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,BaggingRegressor,AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn import metrics
from tqdm import tqdm

from datetime import datetime as dt
import numpy as np
import pandas as pd
import pandas_datareader as web
import plotly.offline as pyo
import plotly.graph_objs as go

###########################################
# HELPER FUNCTIONS
###########################################

# save model using joblib
def save_model(model,name):
    filename = f'models/{name}'
    joblib.dump(model,filename)

def load_model(name):
    filename = f'models/{name}/model.plk'
    return joblib.load(filename)

# generates metrics and puts them in a pandas series
def gen_matrics(y_test,y_pred):
    standard_metrics = pd.Series()

    for arg in vars(args):
        val = getattr(args,arg)
        if val:
            standard_metrics.loc[arg] = val

    standard_metrics.loc['Explained Variance'] = metrics.explained_variance_score(y_test, y_pred)
    standard_metrics.loc['MAE'] = metrics.mean_absolute_error(y_test, y_pred)
    standard_metrics.loc['MSE'] = metrics.mean_squared_error(y_test, y_pred)
    standard_metrics.loc['MedAE'] = metrics.median_absolute_error(y_test, y_pred)
    standard_metrics.loc['RSQ'] = metrics.r2_score(y_test, y_pred)

    return standard_metrics


def make_pipeline_and_grid(steps) :
    """Make composite pipeline and parameter grid from list of estimators.
        You do NOT have to understand the implementation of this function.
        It stitches together the input steps and generates the nested parameter grid.

        Parameters
        ----------
        steps : list
            List of (name, transform) tuples (implementing fit/transform) that are chained,
            in the order in which they are chained, with the last object an estimator.

            Each transform should have either a transformer_ or an estimator_ attribute.
            These attributes store a sklearn object that can transform or predict data.

            Each transform should have a param_grid.
            This attribute stores the hyperparameter grid for the transformer or estimator.
    """
    pipe_steps = []
    pipe_param_grid = {}

    # chain transformers
    for (name, transform) in steps[:-1]:
        transformer = transform.transformer_
        pipe_steps.append((name, transformer))
        for key, val in transform.param_grid_.items():
            pipe_param_grid[name + "__" + key] = val

    # chain estimator
    name, transform = steps[-1]
    estimator = transform.estimator_
    pipe_steps.append((name, estimator))
    for key, val in transform.param_grid_.items():
        pipe_param_grid[name + "__" + key] = val

    # stitch together preprocessors and classifier
    pipe = Pipeline(pipe_steps)
    return pipe, pipe_param_grid

def gen_subplot(train_index,test_index,y_train,y_test,y_pred,legend=False):
    """ generates subplot for a single model

    """
    traces = [
            go.Scatter(
                x = train_index,
                y = y_train,
                line = dict(
                    color = 'blue'
                ),
                showlegend=legend,
                name='Training'
            ),
            go.Scatter(
                x = test_index,
                y = y_test,
                line = dict(
                    color = 'black'
                ),
                showlegend=legend,
                name='Test'
            ),
            go.Scatter(
                x = test_index,
                y = y_pred,
                line = dict(
                    color = 'orange'
                ),
                showlegend=legend,
                name='Prediction'
            )]
    return traces


def plot_results(traces,clf_strs,ticker):
        #train_index,test_index,y_train,y_test,y_pred,clf_strs):
    """ Plots results of various models

        Parameters
        ----------
        train_index: Series
            training data dates
        test_index: Series
            test data dates
        y_train: list
            training labels
        y_test: list
            test labels
        clf_strs:
            Model names

        Returns
        -------
        fig: plotly Figure
            figure with subplot for each model
    """
    # gather plot info
    clf_count = len(clf_strs)
    rows = (clf_count//3 + 1)
    cols = 3
    cur_col = 1
    cur_row = 1

    # figure
    fig = make_subplots(
        subplot_titles=[clf_str for clf_str in clf_strs],
        rows=rows,
        cols=cols
    )

    # add subplots
    for i in range(len(traces)):
        [fig.append_trace(trace,row=cur_row,col=cur_col) for trace in traces[i]]
        cur_col += 1
        if cur_col > cols:
            cur_col = 1
            cur_row +=1

    # add figure title
    fig.update_layout(
        title=dict(
                text=ticker,
                x=0.5,
                xanchor='center',
                yanchor='top'
        )
    )
    return fig

##################################################
#               MAIN                             #
##################################################
def main(tickers):
    np.random.seed(1234)

    # display ticker info
    print("Tickers = ",tickers)

    print()

    # read data
    print("Reading data ... ")
    PREFIX = config.gen_prefix(tickers)

    # relevant filenames
    FEATURES_TRAIN_FILENAME = f'../data/processed/{PREFIX}train_features.csv'
    FEATURES_TEST_FILENAME = f'../data/processed/{PREFIX}test_features.csv'
    LABELS_TRAIN_FILENAME = f'../data/processed/{PREFIX}train_labels.csv'
    LABELS_TEST_FILENAME = f'../data/processed/{PREFIX}test_labels.csv'

    # load data
    X_train = pd.read_csv(FEATURES_TRAIN_FILENAME).set_index('date')
    X_test = pd.read_csv(FEATURES_TEST_FILENAME).set_index('date')
    y_train = pd.read_csv(LABELS_TRAIN_FILENAME).set_index('date')
    y_test = pd.read_csv(LABELS_TEST_FILENAME).set_index('date')

    # make validation dataset



    # reduce number of features
    # NOTE: want to add this step to pipeline
    X_train,X_test = config.reduce_features(X_train,X_test,y_train)
    y_train,y_test = y_train.target.ravel(),y_test.target.ravel()

    # get necessary sizes
    n,d = X_train.shape

    # define cv
    cv_inner = TimeSeriesSplit(n_splits=5)
    cv_outer = TimeSeriesSplit(n_splits=5)

    # get classifier names
    clf_strs = classifiers.CLASSIFIERS

    # scoring dct to track performance
    scores = {}

    # dislays subplot legend
    plot_traces = []
    legend=True

    # run all classifiers on dataset
    for clf_str in clf_strs:
        print("\nclf = ",clf_str)

        dct = {}

        # define pipeline
        clf = getattr(classifiers, clf_str)(n,d)
        steps = [('Scaler',preprocessors.Scaler()),(clf_str,clf)]
        pipe,param_grid = make_pipeline_and_grid(steps)

        # determine which CV to be used
        if clf_str in ['Dummy','LinearRegressor','KNN','lr_boost']:
            search = GridSearchCV(pipe,param_grid,
                                  cv=cv_inner,
                                  refit=True,
                                  iid=False,
                                  scoring='neg_mean_absolute_error',
                                  return_train_score=True,
                                  n_jobs=-1)
        #pipe_scores = cross_val_score(search,X_train,
        #                              y_train,cv=cv_outer
        #                             ,scoring='neg_mean_squared_error').mean()

        else:
            search = RandomizedSearchCV(pipe,param_grid,
                                        cv=cv_inner,
                                        n_iter=20,
                                        iid=False,
                                        random_state=0,
                                        refit=True,
                                        scoring="neg_mean_absolute_error",
                                        return_train_score=True,
                                        n_jobs=-1)
        # training
        print("Training Classifier ... ")
        search.fit(X_train,y_train)
        results = search.cv_results_

        # make predictions
        print("Making Predictions ... ")
        y_pred = search.predict(X_test)

        # generate subplot traces
        subplot_traces = gen_subplot(
            X_train.index,
            X_test.index,
            y_train,
            y_test,
            y_pred,
            legend
        )

        # append subplot to plot
        legend=False
        plot_traces.append(subplot_traces)

        # record metrics
        print("Recording Metrics ... \n")
        test_mar = metrics.mean_absolute_error(y_test,y_pred)
        dct['train_mar'] = -1*(results['mean_train_score'].mean())
        dct['test_mar'] = test_mar
        dct['difference'] = dct['test_mar'] - dct['train_mar']

        # update scores dict
        scores[clf_str] = dct

    # Display results
    scores_df = pd.DataFrame.from_dict(scores,orient='index')
    print(scores_df)
    print("\nBest Classifier: ",scores_df.test_mar.idxmin())
    print("Mean Absolute Error: ",scores_df.test_mar.min())

    # plot results
    print("Plotting Results ... ")
    fig = plot_results(plot_traces,clf_strs,tickers[0])
    fig.show()


if __name__ == '__main__':
    main(sys.argv[1:])
