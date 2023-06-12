#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 20:24:24 2022

@author: vince
"""

import pandas as pd
import numpy as np

from bokeh.layouts import column
from bokeh.models import BoxAnnotation, Paragraph, ColumnDataSource ,DataTable, TableColumn, DateFormatter, HoverTool
from bokeh.plotting import figure, output_file, show

from key_metrics import annual_return, sharpe_ratio, calmar_ratio, win_rate
from key_metrics import drawdown_series, max_drawdown, ranked_ic_series, ranked_ic_stats, turnover



def ranked_hedge_analysis(df0,f,r,n=5,fees=0,leverage=1,slippage=0,compound=True,plot=True,weight=None):
    if weight == None:     
        df=df0[[f,r]].dropna().sort_index()
        df[f]=df.groupby('datetime')[f].apply(lambda x:pd.qcut(x,n,labels=range(n))).astype(int)  
        ret=df[[f,r]].groupby([f,'datetime']).mean().reset_index()
        ret=ret.pivot(index='datetime',columns=f,values=r).fillna(0)
        ret['hedge']=((ret[n-1]-ret[0]-slippage-fees)*leverage).map(lambda x:max(x,-0.1))
    else:
        df=df0[[f,r,weight]].dropna().sort_index()
        df[f]=df.groupby('datetime')[f].apply(lambda x:pd.qcut(x,n,labels=range(n))).astype(int)  
        ret=df[[f,r,weight]].groupby([f,'datetime']).agg(r=(r,lambda x: np.average(x, weights=df.loc[x.index, weight]))).reset_index()
        ret_l=ret.pivot(index='datetime',columns=f,values='r').fillna(0)
        #ret=df[[f,r,weight]].groupby([f,'datetime']).agg(r=(r,lambda x: np.average(x, weights=1/df.loc[x.index, weight]))).reset_index()
        ret=df[[f,r]].groupby([f,'datetime']).mean().reset_index()
        ret=ret.pivot(index='datetime',columns=f,values=r).fillna(0)
        ret['hedge']=((ret_l[n-1]-ret[0]-slippage-fees)*leverage).map(lambda x:max(x,-0.1))
    if compound:
        cum=(ret+1).cumprod()
    else:
        cum=ret.cumsum()+1
    
    total_cum=round((cum["hedge"].iloc[-1]-1)*100,2)
    mean_ret=round(ret.hedge.mean()*100,2)
    dd=round(max_drawdown(ret.hedge,compound)*100,2)
    sharpe=round(sharpe_ratio(ret.hedge),2)
    calmar=round(calmar_ratio(ret.hedge,compound),2)
    win_rate=round((ret.hedge>0).mean()*100,2)
    payoff=round(abs(ret.hedge[ret.hedge>0].mean()/ret.hedge[ret.hedge<0].mean()),2)
    
    tools=['pan','wheel_zoom','box_zoom','reset']
    p=figure(width=800,height=300,x_axis_type='datetime',tools=tools, title=f"Total:{total_cum}%, Sharpe:{sharpe}, Mean Ret:{mean_ret}%, Max DD:{dd}%, Calmar:{calmar}, Win rate:{win_rate}%, Payoff:{payoff}")
    p.line(cum.index,cum['hedge'],color='gold') 
    low_box=BoxAnnotation(top=1,fill_alpha=0.1,fill_color='red')
    high_box=BoxAnnotation(bottom=1,fill_alpha=0.1,fill_color='green')
    p.add_layout(low_box)
    p.add_layout(high_box)
    hover=HoverTool(tooltips=[('Date','@x{%Y-%m-%d %H:%M:%S}'), ('Cumulative Ret','@y')],formatters={'@x':'datetime'})
    p.add_tools(hover)
    if plot:
        show(p) 
    return p,ret

def quick_analysis(df0,f,r,n=5,fees=0,leverage=1,slippage=0,compound=True,weight=None,info=''):
    p0=Paragraph(text=f"""{f} {df0.index.get_level_values('datetime').min().date()} to {df0.index.get_level_values('datetime').max().date()};\n fees:{fees}, leverge:{leverage}, slippage:{slippage}"""+info,width=800)
    p1=ranked_ic_series(df0,f,r,plot=False)
    p2,ret=ranked_hedge_analysis(df0,f,r,n,fees,leverage,slippage,compound,plot=False,weight=weight)
    show(column(p0,p1,p2))
    
def factor_style_analysis(df0,f,fret,pret,n=5):
    df=df0[[f,fret,pret]].dropna().sort_index()
    df[f]=df.groupby('datetime')[f].apply(lambda x:pd.qcut(x,n,labels=range(n))).astype(int)  
    ret=df[[f,fret,pret]].groupby([f]).mean().reset_index()
    return ret
    
    
    
    
    
    