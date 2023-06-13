#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 15:34:25 2022

@author: vince
"""

import numpy as np
import pandas as pd


def demean(s):
    """
    Wrapper function to remove mean from variable.
    :param s: a pandas Series.
    :return: a pandas Series with mean being removed.
    """
    df=pd.merge(s,s.groupby('datetime').agg(['mean']),how='left',left_index=True,right_index=True)
    return df[df.columns[0]] - df[df.columns[1]]

def standardize(s):
    """
    Wrapper function to standardize varible.
    :param s: a pandas Series.
    :return: a pandas Series with the time-series rank over the past window days.
    """
    df=pd.merge(s,s.groupby('datetime').agg(['mean','std']),how='left',left_index=True,right_index=True)
    return (df[df.columns[0]]- df[df.columns[1]])/df[df.columns[2]]


def signedpower(s, power = 2):
    return (np.sign(s)*np.power(s, power)).sort_index()

def ts_rank(df, window=10):
    """
    Wrapper function to estimate rolling rank.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series rank over the past window days.
    """
    return df.groupby('ticker').rolling(window).rank(pct=True).droplevel(0).sort_index()

def ts_min(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.groupby('ticker').rolling(window).min().droplevel(0).sort_index()

def ts_max(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series max over the past 'window' days.
    """
    return df.groupby('ticker').rolling(window).max().droplevel(0).sort_index()

def ts_mean(df, window=10):
    """
    Wrapper function to estimate SMA.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series mean over the past 'window' days.
    """
    return df.groupby('ticker').rolling(window).mean().droplevel(0).sort_index()

def ts_std(df, window=10):
    """
    Wrapper function to estimate rolling standard deviation.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series standard deviation over the past 'window' days.
    """
    return df.groupby('ticker').rolling(window).std().droplevel(0).sort_index()

def ts_skew(df, window=10):
    """
    Wrapper function to estimate rolling sum.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series skewness over the past 'window' days.
    """
    
    return df.groupby('ticker').rolling(window).skew().droplevel(0).sort_index()

def ts_kurtosis(df, window=10):
    """
    Wrapper function to estimate rolling sum.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series kutosis over the past 'window' days.
    """
    
    return df.groupby('ticker').rolling(window).kurtosis().droplevel(0).sort_index()

def ts_cor(x, y, window=10):
    """
    Wrapper function to estimate rolling corelations.
    :param x: a pandas Series.
    :param y: a pandas Series.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series correlation over the past 'window' days.
    """
    #return x.groupby('ticker').rolling(window).corr(y).sort_index()
    s = pd.concat([x,y], axis = 1)
    return s.reset_index().set_index('datetime').groupby('ticker').rolling(window).corr().droplevel(2).iloc[::2, 1].sort_index()

def ts_cov(x, y, window=10):
    """
    Wrapper function to estimate rolling covariance.
    :param x: a pandas Series.
    :param y: a pandas Series.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series covariance over the past 'window' days.
    """
    s = pd.concat([x,y], axis = 1)
    return s.reset_index().set_index('datetime').groupby('ticker').rolling(window).cov().droplevel(2).iloc[::2, 1].sort_index()

def ts_sum(df, window=10):
    """
    Wrapper function to estimate rolling sum.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series sum over the past 'window' days.
    """
    
    return df.groupby('ticker').rolling(window).sum().droplevel(0).sort_index()

def ts_diff(df, period=1):
    """
    Wrapper function to estimate difference.
    :param df: a pandas DataFrame.
    :param period: the difference grade.
    :return: a pandas DataFrame with today’s value minus the value 'period' days ago.
    """
    return df.groupby('ticker').diff(int(period)).sort_index()

def rolling_prod(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The product of the values in the array.
    """
    return np.prod(na)

def ts_prod(df, window=10):
    """
    Wrapper function to estimate rolling product.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series product over the past 'window' days.
    """
    return df.groupby('ticker').rolling(window).apply(rolling_prod).droplevel(0).sort_index()

def ts_argmax(df, window=10):
    """
    Wrapper function to estimate which day ts_max(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.groupby('ticker').rolling(window).apply(np.argmax).droplevel(0).sort_index() + 1 

def ts_argmin(df, window=10):
    """
    Wrapper function to estimate which day ts_min(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.groupby('ticker').rolling(window).apply(np.argmin).droplevel(0).sort_index() + 1

def ts_delay(df, period=1):
    """
    Wrapper function to estimate lag.
    :param df: a pandas DataFrame.
    :param period: the lag grade.
    :return: a pandas DataFrame with lagged time series
    """
    return df.groupby('ticker').shift(period)

def rank(df):
    """
    Cross sectional rank
    :param df: a pandas DataFrame.
    :group: cross section group
    :return: a pandas DataFrame with rank along columns.
    """
    #return df.rank(axis=1, pct=True)
    return df.groupby('datetime').rank(pct=True).sort_index()

def scale(df, center=1):
    """
    Scaling time serie.
    :param df: a pandas DataFrame.
    :param k: scaling factor.
    :return: a pandas DataFrame rescaled df such that sum(abs(df)) = k
    """
    df = df.replace([np.inf, -np.inf], [np.nan, np.nan])
    return (df / np.abs(df).groupby('datetime').transform('sum') * center).sort_index()


def decay_linear(df, period=10):
    """
    Linear weighted moving average implementation.
    :param df: a pandas DataFrame.
    :param period: the LWMA period
    :return: a pandas DataFrame with the LWMA.
    """
    # Clean data
    if df.isnull().values.any():
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(value=0, inplace=True)
    na_lwma = np.zeros_like(df)
    na_lwma[:period, :] = df.iloc[:period, :] 
    na_series = df.values # changed here

    divisor = period * (period + 1) / 2
    y = (np.arange(period) + 1) * 1.0 / divisor
    # Estimate the actual lwma with the actual close.
    # The backtest engine should assure to be snooping bias free.
    for row in range(period - 1, df.shape[0]):
        x = na_series[row - period + 1: row + 1, :]
        na_lwma[row, :] = (np.dot(x.T, y))
    return pd.DataFrame(na_lwma, index=df.index)

def rolling_regression(df,x,y,regressor,reg_period,reg_period_unit='D'):
    """
    Rolling regression implementation.
    :param df: a pandas DataFrame.
    :param x: independent variable
    :param y: depedent variable
    :param regressor: regressor i.e. linear regression from sklearn or statsmodel
    :param reg_period: regression rolling period
    :param reg_period_unit: unit of regression period, i.e. 'D', 'h', 'm', 's'
    :return: a pandas DataFrame with rolling regression predicted value
             and a pandas DataFrame with regression coeficients.
    """
    prediction=pd.DataFrame()
    coef=[]
    for dt in df.index.unique(level='datetime').sort_values():
        if dt-np.timedelta64(reg_period, reg_period_unit)>=min(df.index.get_level_values('datetime')):
            train=df[(df.index.get_level_values('datetime')>=(dt-np.timedelta64(reg_period, reg_period_unit)))&(df.index.get_level_values('datetime')<dt)]
            regressor.fit(train[x],train[y])
            coef.append(list(regressor.coef_))
            temp=df[df.index.get_level_values('datetime')==dt].copy()
            temp['model']=regressor.predict(temp[x])
            prediction=pd.concat([prediction,temp])
    return prediction,pd.DataFrame(coef,columns=x)