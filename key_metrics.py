#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 20:24:24 2022

@author: vince
"""

import numpy as np
import pandas as pd
from scipy import stats

def annual_return(ret,trading_days=365):
    s=(ret+1).cumprod()      
    total_ret=s[-1]-1
    num_years=(ret.index.max()-ret.index.min()).days/trading_days
    annual_return = (1. + total_ret) ** (1. / num_years) - 1
    return annual_return

def sharpe_ratio(ret,trading_days=365):
    if np.std(ret)!=0:
        return np.mean(ret)/np.std(ret)*np.sqrt(trading_days)  
    else:
        return np.mean(ret)*np.sqrt(trading_days)  

def calmar_ratio(ret):
    max_dd = max_drawdown(ret)
    if max_dd < 0:
        calmar = annual_return(ret) / abs(max_dd)
        if np.isinf(calmar):
            return np.nan
        else:
            return calmar
    else:
        return np.nan

def win_rate(ret,benchmark=0):
    win=((ret-benchmark)>0).sum()
    loss=((ret-benchmark)<0).sum()
    return win/(win+loss)
    
def drawdown_series(ret):
    s=(ret+1).cumprod()
    cummax=s.cummax()
    return s/cummax-1

def max_drawdown(ret):
    s=(ret+1).cumprod()
    cummax=s.cummax()
    return (s/cummax-1).min()

def ic_series(df0,f1,f2):
    df=df0[[f1,f2]].dropna().sort_index()
    df=df.groupby('datetime').apply(lambda x:x[f1].corr(x[f2]))
    return df.dropna()
    
def ranked_ic_series(df0,f1,f2):
    df=df0[[f1,f2]].dropna().sort_index()
    df[f1]=df.groupby('datetime')[f1].rank()
    df[f2]=df.groupby('datetime')[f2].rank()
    df=df.groupby('datetime').apply(lambda x:x[f1].corr(x[f2]))
    return df.dropna()

def ic_stats(df):
    tStat,pValue = stats.ttest_1samp(df,0)
    ic_mean=df.mean()
    icir=df.mean()/df.std()
    return ic_mean, icir, tStat, pValue

def turnover(l_candid,l_cur,s_candid=None,s_cur=None):
    total=len(l_cur)+len(s_cur)
    close_position=0
    for i in l_cur:
        if i not in l_candid:
            close_position += 1
    for i in s_cur:
        if i not in s_candid:
            close_position += 1
    open_position=0
    for i in l_candid:
        if i not in l_cur:
            open_position += 1
    for i in s_candid:
        if i not in s_cur:
            open_position += 1
    return 0.5*(close_position+open_position)/total

def style_detect(df0,f,fret,pret,n=5):
    df=df0[[f,fret,pret]].dropna().sort_index()
    df[f]=df.groupby('datetime')[f].apply(lambda x:pd.qcut(x,n,labels=range(n))).astype(int)  
    ret=df[[f,fret,pret]].groupby([f]).mean().reset_index()
    return ret

    
    
    
    
    
    