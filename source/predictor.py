#!/usr/bin/env python3
# Author: Jeremy Jess

import pandas as pd
import numpy as np
import warnings
import joblib
import sys
import config
import models
import preprocessors

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

from plotly.subplots import make_subplots
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer,explained_variance_score,mean_absolute_error,mean_squared_error,r2_score

import plotly.graph_objs as go

###########################################
# Global Vars
###########################################
METRICS = {'explained_variance_score':make_scorer(explained_variance_score),
           'mean_absolute_error':make_scorer(mean_absolute_error),
           'mean_squared_error':make_scorer(mean_squared_error),
           'r2_score':make_scorer(r2_score)}

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
def gen_metrics(y_test,y_pred,dataset='test'):
    standard_metrics = {}

    standard_metrics[f'{dataset}_explained_variance_score'] = explained_variance_score(y_test, y_pred)
    standard_metrics[f'{dataset}_mean_absolute_error'] = mean_absolute_error(y_test, y_pred)
    standard_metrics[f'{dataset}_mean_squared_error'] = mean_squared_error(y_test, y_pred)
    standard_metrics[f'{dataset}_r2_score'] = r2_score(y_test,y_pred)

    return standard_metrics

def gen_scorer(scoring_metrics):
    """Generates dict with specified metrics for CV

        Parameters
        -----------
        scoring_metrics: list
            list of metrics to evaluate during CV

        Returns
        --------
        scoring_dict: dict
            dict with scoring metrics
    """
    scoring_dict = {}
    for metric in scoring_metrics:
        scoring_dict[metric] = make_scorer(metric)
    return scoring_dict

def split_train_test(X,y,train_percent=0.7):
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
        split_index = int(X.shape[0]*train_percent)

        X_train = X[0:split_index]
        y_train = y[0:split_index]

        X_test = X[split_index:]
        y_test = y[split_index:]

        return X_train,X_test,y_train,y_test

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

        Parameters
        -----------
        train_index: Series
                training data dates
        test_index: Series
            test data dates
        y_train: list
            training labels
        y_test: list
            test labels
        y_pred: list
            predicted labels
        legend: bool
            indicates whether to include legend in plot

        Returns
        -------
        traces: list
            list of traces for subplot
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


def plot_results(traces,model_strs,ticker):
        #train_index,test_index,y_train,y_test,y_pred,model_strs):
    """ Plots results of various models

        Parameters
        ----------
        traces: list
            list of subplot traces
        model_strs:
            Model names

        Returns
        -------
        fig: plotly Figure
            figure with subplot for each model
    """
    # gather plot info
    clf_count = len(model_strs)
    rows = (clf_count//3 + 1)
    cols = 3
    cur_col = 1
    cur_row = 1

    # figure
    fig = make_subplots(
        subplot_titles=[model_str for model_str in model_strs],
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
def main(argv):
    np.random.seed(1234)

    if len(argv) != 3:
        print("Must be in format: python featurize.py  <TICKER> <FORWARD_LAG>")
        exit(0)
    elif not int(argv[2]):
        print("Must be in format: python featurize.py  <TICKER> <FORWARD_LAG>")
        exit(0)

    # set relevant vars
    ticker = argv[1]
    forward_lag  = int(argv[2])

    # display ticker info
    print("Ticker = ",ticker)
    print(f"Prediction Window = {forward_lag} days")

    print()

    # read data
    print("Reading data ... ")
    PREFIX = config.gen_prefix(ticker,forward_lag)

    # relevant filenames
    """
    FEATURES_TRAIN_FILENAME = f'../data/processed/{PREFIX}_train_features.csv'
    FEATURES_TEST_FILENAME = f'../data/processed/{PREFIX}_test_features.csv'
    LABELS_TRAIN_FILENAME = f'../data/processed/{PREFIX}_train_labels.csv'
    LABELS_TEST_FILENAME = f'../data/processed/{PREFIX}_test_labels.csv'
    """

    FEATURES_FILENAME = f'../data/processed/{PREFIX}_features.csv'
    LABELS_FILENAME = f'../data/processed/{PREFIX}_labels.csv'

    # load data
    X = pd.read_csv(FEATURES_FILENAME).set_index('date')
    y = pd.read_csv(LABELS_FILENAME).set_index('date')

    """
    X_train = pd.read_csv(FEATURES_TRAIN_FILENAME).set_index('date')
    X_test = pd.read_csv(FEATURES_TEST_FILENAME).set_index('date')
    y_train = pd.read_csv(LABELS_TRAIN_FILENAME).set_index('date')
    y_test = pd.read_csv(LABELS_TEST_FILENAME).set_index('date')
    """

    # Split into train and test data
    X_train,X_test,y_train,y_test = split_train_test(X,y,train_percent=0.7)

    # relevant dates
    print(f"Training data range:\n\t {str(X_train.index[0])[:10]} to {str(X_train.index[-1])[:10]}")
    print(f"Test data range:\n\t {str(X_test.index[0])[:10]} to {str(X_test.index[-1])[:10]}")

    print()

    # reduce number of features
    # NOTE: want to add this step to pipeline
    X_train,X_test = config.reduce_features(X_train,X_test,y_train)
    y_train,y_test = y_train.target.ravel(),y_test.target.ravel()

    # get necessary sizes
    n,d = X_train.shape

    # define cv
    cv_inner = TimeSeriesSplit(n_splits=5)
    #cv_outer = TimeSeriesSplit(n_splits=5)


    # get classifier names
    model_strs = models.MODELS

    # scoring dct to track performance
    scores = {}

    # dislays subplot legend
    plot_traces = []
    legend=True

    # run all models on dataset
    for model_str in model_strs:
        print("\nmodel = ",model_str)

        dct = {}

        # define pipeline
        model = getattr(models, model_str)(n,d)
        steps = [('Scaler',preprocessors.Scaler()),(model_str,model)]
        pipe,param_grid = make_pipeline_and_grid(steps)

        # determine which CV to be used
        if model_str in ['Dummy','LinearRegressor','KNN','lr_boost']:
            search = GridSearchCV(pipe,param_grid,
                                  cv=cv_inner,
                                  refit='mean_absolute_error',
                                  iid=False,
                                  scoring=METRICS,
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
                                        refit='mean_absolute_error',
                                        scoring=METRICS,
                                        return_train_score=True,
                                        n_jobs=-1)
        # training
        print("Training Model ... ")
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

        for met in METRICS:
            dct[f'train_{met}'] = results[f'mean_train_{met}'].mean()

        test_scores = gen_metrics(y_test,y_pred)
        dct.update(test_scores)

        # update scores dict
        scores[model_str] = dct

    # Display results
    scores_df = pd.DataFrame.from_dict(scores,orient='index')
    print(scores_df)
    print("\nBest Model: ",scores_df.test_mean_squared_error.idxmin())
    print("Mean Squared Error: ",scores_df.test_mean_squared_error.min())

    print()

    # show predictions
    # NOTE: this is hacky, need to fix later
    print(f'{forward_lag} day predictions')
    RAW_PREFIX = PREFIX.replace(f'_{forward_lag}','')
    raw_df = pd.read_csv(f'../data/raw/{RAW_PREFIX}_hist.csv').set_index('date')
    prediction_dates = raw_df.index[-forward_lag:]
    for i,date in enumerate(prediction_dates):
        print(f'{date}: {y_pred[-forward_lag+i]:.2f}')

    print()

    # plot results
    print("Plotting Results ... ")
    #fig = plot_results(plot_traces,model_strs,ticker)
    #fig.show()

    print()

    # log results
    print("Saving Results ... ")
    RESULTS_FILENAME = f'../data/results/{PREFIX}.csv'
    scores_df.to_csv(RESULTS_FILENAME)
    print(f'\{RESULTS_FILENAME}')

if __name__ == '__main__':
    main(sys.argv)
