from utils import in_notebook, utcToChi, getChiTimeNow, readCSV
from combinations import getCombDF
from global_vars import omni_dir, tdm_dir, day_chg_incs, minute_incs, keras_models
from train import autoTrain

import pandas as pd
import numpy as np

# omni_dir = '/home/andrew/All_Trading/Studies/Omni_Project/'
# # config_dir = omni_dir + 'Primary_Assets/US_Dollar_Index/Config1/'
# tdm_dir = '/media/andrew/FreeAgent Drive/Market_Data/Tick_Data_Manager/'
regression_results_fn = '/media/andrew/FreeAgent Drive/Market_Data/Tick_Data_Manager/regression_results.csv'
data_summary_fn = tdm_dir+'data_summary.csv'
data_summaryDF = pd.read_csv(data_summary_fn)

sec = 'KELLOGG_CO'

combDF = getCombDF(regression_results_fn, data_summary_fn)
# combDF.to_csv(omni_dir+'combDF_temp.csv', index=False)
# combDF = readCSV(omni_dir+'combDF_temp.csv')

autoTrain(sec, combDF, keras_models, nrows=15000)
