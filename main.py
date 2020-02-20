from utils import in_notebook, utcToChi, getChiTimeNow, readCSV, autoChooseSecurity
from combinations import getCombDF
from global_vars import omni_dir, tdm_dir, day_chg_incs, minute_incs, keras_models
from train import autoTrain

import pandas as pd
import numpy as np
import sys, os

# sec = 'KELLOGG_CO'
sec = 'NONE CHOSEN'

print(sys.argv)
if len(sys.argv)==1:
    print('No sec argument, autoChoosing.')
    sec = autoChooseSecurity(omni_dir)
else:
    sec = sys.argv[1]
    print(sec, 'read as sec in main.py')

# omni_dir = '/home/andrew/All_Trading/Studies/Omni_Project/'
# # config_dir = omni_dir + 'Primary_Assets/US_Dollar_Index/Config1/'
# tdm_dir = '/media/andrew/FreeAgent Drive/Market_Data/Tick_Data_Manager/'
regression_results_fn = tdm_dir+'regression_results.csv'
data_summary_fn = tdm_dir+'data_summary.csv'
data_summaryDF = pd.read_csv(data_summary_fn)

combDF = getCombDF(regression_results_fn, data_summary_fn)
# combDF.to_csv(omni_dir+'combDF_temp.csv', index=False)
# combDF = readCSV(omni_dir+'combDF_temp.csv')

# record our security choice to the work log
work_log_fn = omni_dir + 'work_log.csv'
work_logDF = pd.DataFrame(columns=['Sec', 'ChosenTime'])
if not os.path.exists(work_log_fn):
    print('work log doesnt exist. Creating a new one:', work_log_fn)
else:
    work_logDF = pd.read_csv(work_log_fn, parse_dates=['ChosenTime'])
work_logDF = work_logDF.append({'Sec': sec, 'ChosenTime': pd.datetime.now()}, ignore_index=True)

print('='*30, '\nsaving work_logDF to', work_log_fn)
print(work_logDF.tail())

work_logDF.to_csv(work_log_fn, index=False)

autoTrain(sec, combDF, keras_models, nrows=50000)
