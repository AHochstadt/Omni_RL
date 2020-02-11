import keras.backend as K
import tensorflow as tf
import numpy as np
import pandas as pd

def getLikelyTickSize(priceDF):
    """Find the first column that has type float and use that"""
    print('Finding likely tick size')
    cols = priceDF.columns
    for col in cols:
        if priceDF[col].dtype == float:
            print('Using', col, 'to determine tick size')
            abs_diffs = priceDF[col][:10000].diff().abs()
            nonzero_abs_diffs = abs_diffs.loc[abs_diffs > 0]
            likely_tick_size = nonzero_abs_diffs.min()
            print('likely_tick_size =', likely_tick_size)
            return likely_tick_size
    raise ValueError('No columns have type float. Cannot retrieve likely tick size.')


def huber_loss(y_true, y_pred, clip_delta=1.0):
    """Huber loss - Custom Loss Function for Q Learning

    Links: 	https://en.wikipedia.org/wiki/Huber_loss
            https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
    """
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta
    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
    return K.mean(tf.where(cond, squared_loss, quadratic_loss))

def sigmoid(x):
    """Performs sigmoid operation
    """
    try:
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))
    except Exception as err:
        print("Error in sigmoid: " + err)

def calcOpenPnl(agent, dataObj, t):
    last_t_midpt = sum(dataObj.train_close_pricesDF[['C_B1', 'C_A1']].iloc[t-1].values)/2
    t_midpt = sum(dataObj.train_close_pricesDF[['C_B1', 'C_A1']].iloc[t].values)/2
    return (agent.position * (t_midpt - last_t_midpt))

def getMaxDraw(pnl_series, direction):
    direction = direction.lower()
    assert direction in ['down', 'up'], 'invalid direction: '+str(direction)
    cum_pnl = pnl_series.cumsum()
    result = 0
    if direction == 'down':
        highest_peak = 0
        for c in cum_pnl.values:
            result = min(result, c-highest_peak)
            highest_peak = max(highest_peak, c)
    if direction == 'up':
        lowest_trough = 0
        for c in cum_pnl.values:
            result = max(result, c-lowest_trough)
            lowest_trough = min(lowest_trough, c)
    return result

def calcSharpe(trade_log):
#     daily_log = trade_log.groupby(trade_log.Minute.str[:10]).agg({'TotalPNL': 'sum'})
    daily_log = trade_log.groupby(trade_log.Minute.dt.date).agg({'TotalPNL': 'sum'})

    trading_days = 252
    return np.sqrt(trading_days)*(daily_log.TotalPNL.mean()/daily_log.TotalPNL.std())

def getTradeLogStats(agent):
    trade_log = agent.trade_log.copy()
    trade_log['Midpt'] = (trade_log.C_B + trade_log.C_A)/2
    trade_log['DayPNL'] = (trade_log.ActualAction!=0)*(trade_log.C_B - trade_log.Midpt) #it's always negative half the bid-ask spread
    trade_log['OpenPNL'] = trade_log.NewPosition.shift() * (trade_log.Midpt - trade_log.Midpt.shift())
    trade_log.OpenPNL.loc[0] = 0
    trade_log['TotalPNL'] = trade_log.DayPNL + trade_log.OpenPNL

    pnl = trade_log.TotalPNL.sum()
#     num_days = len(trade_log.Minute.str[:10].unique())
    num_days = len(trade_log.Minute.dt.date.unique())
    pnl_per_day = pnl/num_days
    sharpe = calcSharpe(trade_log)
    num_trades = (trade_log.ActualAction!=0).sum()
    drawdown = getMaxDraw(trade_log.TotalPNL, 'down')
    drawup = getMaxDraw(trade_log.TotalPNL, 'up')
    ts_seen = len(trade_log)
    pct_flat = (trade_log.NewPosition == 0).sum()/ts_seen
    pct_long = (trade_log.NewPosition == 1).sum()/ts_seen
    pct_short = (trade_log.NewPosition == -1).sum()/ts_seen
    data_start = trade_log.Minute.dt.date.iloc[0]
    data_end = trade_log.Minute.dt.date.iloc[-1]

    return pnl, pnl_per_day, sharpe, num_trades, drawdown, drawup, ts_seen, pct_flat, pct_long, pct_short, data_start, data_end

def getSprCrossInfo(agent, dataObj):

    trade_log = agent.trade_log.copy()
    assert (dataObj.train_minutes.iloc[0] <= trade_log.Minute.iloc[-1] <= dataObj.train_minutes.iloc[-1]
            or dataObj.val_minutes.iloc[0] <= trade_log.Minute.iloc[0] <= dataObj.val_minutes.iloc[-1]), trade_log.Minute
    train_or_val = 'train'

    dataDF = dataObj.trainDF
    minutes = dataObj.train_minutes
    close_pricesDF = dataObj.train_close_pricesDF

    if trade_log.Minute.iloc[0] in dataObj.val_minutes.values:
        train_or_val = 'val'
        dataDF = dataObj.valDF
        minutesDF = dataObj.val_minutes
        close_pricesDF = dataObj.val_close_pricesDF

    # get large spread threshold
    avg_spr = (close_pricesDF.C_A1 - close_pricesDF.C_B1).mean()
    spr_quantile = (close_pricesDF.C_A1 - close_pricesDF.C_B1).quantile(q=.8)
    lrg_spr_threshold = max(avg_spr*5, spr_quantile)

    # get stats
    lrg_spr_trade_log = trade_log.loc[(trade_log.C_A - trade_log.C_B) > lrg_spr_threshold]
    if len(lrg_spr_trade_log) == 0:
        return np.nan, np.nan, np.nan

    lrg_spr_minutes = lrg_spr_trade_log.Minute
    lrg_spr_cross_pct = (lrg_spr_trade_log.ActualAction != 0).sum()/len(lrg_spr_trade_log)

    t_idx = minutes.loc[minutes.isin(lrg_spr_minutes)].index
    q_value_array = [agent.getQValues(dataObj, t) for t in t_idx]
    lrg_spr_cross_likelihood = np.mean([(q[0][0] != max(q[0]))  for q in q_value_array]) #needs q[0] because output is wrapped in unnecessary list
    lrg_spr_hold_preference = np.mean([q[0][0]-max(q[0][1:]) for q in q_value_array])

    return lrg_spr_cross_pct, lrg_spr_cross_likelihood, lrg_spr_hold_preference
