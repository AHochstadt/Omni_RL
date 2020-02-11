# utils.py

# import os
# import math
# import logging

# import pandas as pd
# import numpy as np

# import keras.backend as K


# Formats Position
format_position = lambda price: ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))


# Formats Currency
format_currency = lambda price: '${0:.2f}'.format(abs(price))


def show_train_result(result, val_position, initial_offset):
    """ Displays training results
    """
    if val_position == initial_offset or val_position == 0.0:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: USELESS  Train Loss: {:.4f}'
                     .format(result[0], result[1], format_position(result[2]), result[3]))
    else:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: {}  Train Loss: {:.4f})'
                     .format(result[0], result[1], format_position(result[2]), format_position(val_position), result[3],))


def show_eval_result(model_name, profit, initial_offset):
    """ Displays eval results
    """
    if profit == initial_offset or profit == 0.0:
        logging.info('{}: USELESS\n'.format(model_name))
    else:
        logging.info('{}: {}\n'.format(model_name, format_position(profit)))


# def get_stock_data(stock_file):
#     """Reads stock data from csv file
#     """
#     df = pd.read_csv(stock_file)
#     return list(df['Adj Close'])


def switch_k_backend_device():
    """ Switches `keras` backend from GPU to CPU if required.
    Faster computation on CPU (if using tensorflow-gpu).
    """
    if K.backend() == "tensorflow":
        logging.debug("switching to TensorFlow for CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



# main.py


# [] agent.py
class Agent:
    """ Stock Trading Bot """
    def __init__(self, model_cfg, config_dir, strategy="t-dqn", reset_every=200, searchForPretrained=True,
                 action_size=3, cross_q_prem=0, flat_q_prem=0)

# [] models.py
def getHTMLIDStrFromModelStr(model_str)
class ModelConfig:    def __init__(self, input_size, model_array=None, model_str=None, actions_size=3)

# [] data.py
def createConfigDir(comb_row, sec1_dir, day_chg_incs, minute_incs)
def locateConfigDir(comb_row, omni_dir, day_chg_incs, minute_incs)
def getExpectedPostprocessedCols(num_secs, day_chg_incs, minute_incs)
def getTrainValDates(config_dir)
def postprocessData(all_minutesDF, dailyDF, sec_guideDF, config_dir, day_chg_incs, minute_incs, saveProcessed=True)
def loadAndProcessData(comb_row, config_dir, day_chg_incs, minute_incs, minute_dir=tdm_dir+'Minute_Files/')
def processData(comb_row, config_dir, day_chg_incs, minute_incs)
class DataObj:
    def __init__(self, config_dir,
                 train_close_pricesDF=None, trainDF=None, val_close_pricesDF=None, valDF=None,
                 load_val=True, nrows=None)


# [] train.py
def endEpisode(agent, dataObj, batch_size, epsilon_start, avg_loss_array, sess_type, start_time)
def train_model(agent, dataObj, ep_count=20, batch_size=32, evaluate_every=5)
def evaluate_model(agent, dataObj, debug=False, batch_size=32)
def modelHasPromise(model_str, config_dir, exhaustiveness_level)
def chooseStrategy(strategies, config_dir, model_str)
def get_state(agent, dataObj, t)
def evaluateAction(action, agent, dataObj, t)
def autoTrain(sec, combDF=combDF, nrows=None, exhaustiveness_level=2, keras_models=keras_models, omni_dir=omni_dir,
              day_chg_incs=day_chg_incs, minute_incs=minute_incs, email_progress=False)


# [x] utils.py
def in_notebook()
def utcToChi(utc_dt)
def getChiTimeNow()
def readCSV(filename, nrows=None)

# [x] methods.py 
def getLikelyTickSize(priceDF)
def huber_loss(y_true, y_pred, clip_delta=1.0)
def sigmoid(x)
def calcOpenPnl(agent, dataObj, t)
def getMaxDraw(pnl_series, direction)
def calcSharpe(trade_log)
def getTradeLogStats(agent)
def getSprCrossInfo(agent, dataObj)

# [] save_results.py
def createPnlVisPlot(config_dir, desired_vis_plot_fn, ts_window_size=5000)
def plotProgSum(config_dir, model_str, model_progDF, x_col, y_col, scale_col,
                x_label=None, y_label=None, y_colors=None, zero_hline=True, max_point_size=60)
def createModelProgressSummary(config_dir, model_str)
def findMod(bigger_soup, elem_type, search_string)
def updateSummarySheet(config_dir)
def saveProgressSummary(agent, dataObj, batch_size, epsilon_start, epsilon_end, avg_loss_array, session_type, start_time, end_time)
