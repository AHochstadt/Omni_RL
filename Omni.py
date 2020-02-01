#!/usr/bin/env python
# coding: utf-8

# In[1]:


# [x] Determine which combinations of data to try
# [] getData fxn
# [x] Determine which sets of dates to try
# [] Determine types of models to try
# [] Try models
# [] Save results
# [] Use reinforcement learning to come up with bots (?)


# <h1> Initial Setup </h1>

# In[2]:


import pandas as pd
import numpy as np
import os, pytz, tqdm, datetime, gc

import statsmodels.api as sm

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model

from bs4 import BeautifulSoup as Soup

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


omni_dir = '/home/andrew/All_Trading/Studies/Omni_Project/'
tdm_dir = '/media/andrew/FreeAgent Drive/Market_Data/Tick_Data_Manager/'
regression_results_fn = '/media/andrew/FreeAgent Drive/Market_Data/Tick_Data_Manager/regression_results.csv'
data_summaryDF = pd.read_csv(tdm_dir+'data_summary.csv')


# In[4]:


def utcToChi(utc_dt):
    chi_tz = pytz.timezone('America/Chicago')
    chi_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(chi_tz)
    return chi_tz.normalize(chi_dt)
def getChiTimeNow():
    utc_dt = pd.datetime.utcnow()
    return(utcToChi(utc_dt))
getChiTimeNow().strftime(format='%Y-%m-%d %H:%M:%S')


# <h1> Determine which combinations of data to try </h1>

# In[5]:


# load prepare regressionDF
regression_cols = ['Sec1', 'Sec2', 'R2', 'p_value', 'NumDatapoints', 'StartDate', 'EndDate']

regressionDF = pd.read_csv(regression_results_fn)
for col in regressionDF.columns:
    if 'Unnamed' in col:
        regressionDF.drop(columns=[col], inplace=True)

assert(set(regression_cols) == set(regressionDF.columns))
regressionDF = regressionDF[regression_cols]

print(str(len(regressionDF.loc[regressionDF.R2 == 'not found']))+' rows have R2 not found:\n')
print(regressionDF.loc[regressionDF.R2 == 'not found'][['Sec1', 'Sec2']])

regressionDF = regressionDF.loc[regressionDF.R2 != 'not found'].reset_index(drop=True)
regressionDF.R2 = regressionDF.R2.astype(float)
regressionDF


# In[6]:


r2_threshold = .2
currency_r2_mult = 3 #give currencies a boost because a priori I think they should matter
max_sec_bundle = 8 #sec1 can have <= max_sec_bundle other secs in its bundle
min_sec_bundle = 3 #sec1 must have >= max_sec_bundle other secs in its bundle

regressionDF['Asset1'] = 'Security'
regressionDF['Asset2'] = 'Security'
regressionDF.loc[regressionDF.Sec1.isin(['Natural_Gas', 'US_Brent_Crude_Oil', 'US_Dollar_Index', 'US_Light_Crude_Oil']), 'Asset1'] = 'Commodity'
regressionDF.loc[regressionDF.Sec2.isin(['Natural_Gas', 'US_Brent_Crude_Oil', 'US_Dollar_Index', 'US_Light_Crude_Oil']), 'Asset2'] = 'Commodity'
currencies = ['AUD',  'CAD',  'CHF',  'EUR',  'GBP',  'NZD',  'SGD',  'TRY',  'USD',  'ZAR',  'NOK',  'SEK',  'PLN',  'JPY']
regressionDF.loc[(regressionDF.Sec1.str[:3].isin(currencies)) & (regressionDF.Sec1.str[-3:].isin(currencies)), 'Asset1'] = 'Currency'
regressionDF.loc[(regressionDF.Sec2.str[:3].isin(currencies)) & (regressionDF.Sec2.str[-3:].isin(currencies)), 'Asset2'] = 'Currency'

# apply currency_r2_mult where only one asset is a currency
regressionDF['Adjusted_R2'] = regressionDF.R2 * currency_r2_mult**(((regressionDF.Asset1 == 'Currency') & (regressionDF.Asset2 != 'Currency')) | ((regressionDF.Asset1 != 'Currency') & (regressionDF.Asset2 == 'Currency')))

# [] identify and organize funds, ETFs, ETNs, Indexes, etc

comb_cols = ['Sec1']+['Sec'+str(i) for i in range(2,(max_sec_bundle+2))]     +['R2_'+str(i) for i in range(2,(max_sec_bundle+2))]     +['p_value'+str(i) for i in range(2,(max_sec_bundle+2))]     +['StartDate'+str(i) for i in range(2,(max_sec_bundle+2))]     +['EndDate'+str(i) for i in range(2,(max_sec_bundle+2))]     +['NumDatapoints'+str(i) for i in range(2,(max_sec_bundle+2))]

combDF = pd.DataFrame(columns = comb_cols)

regressionDF.loc[regressionDF.R2 >= r2_threshold]


# In[7]:


currencyDF = regressionDF.loc[(regressionDF.Asset1 == 'Currency') | (regressionDF.Asset2 == 'Currency') ].reset_index(drop=True)
currencyDF.sort_values(by='R2', ascending=False)#.head(20)


# In[8]:


# construct combDF
for sec in regressionDF.Sec1.unique():
    sec_sub = regressionDF.loc[regressionDF.Sec1 == sec]
    # [] subset for EndDate
    r2_sub = sec_sub.loc[sec_sub.Adjusted_R2 > r2_threshold]
    final_sub = 'not set yet'
    if len(r2_sub) < min_sec_bundle: #just grab the best min_sec_bundle sec2s
        final_sub = sec_sub.sort_values(by='Adjusted_R2', ascending=False).reset_index(drop=True).iloc[:min_sec_bundle]
    else:
        final_sub = r2_sub.sort_values(by='Adjusted_R2', ascending=False).reset_index(drop=True).iloc[:max_sec_bundle]
    new_comb_row = {'Sec1': sec}
    for i in range(len(final_sub)):
        new_comb_row['Sec'+str(i+2)] = final_sub.Sec2.iloc[i]
        new_comb_row['R2_'+str(i+2)] = final_sub.R2.iloc[i]
        new_comb_row['p_value'+str(i+2)] = final_sub.p_value.iloc[i]
        new_comb_row['StartDate'+str(i+2)] = final_sub.StartDate.iloc[i]
        new_comb_row['EndDate'+str(i+2)] = final_sub.EndDate.iloc[i]
        new_comb_row['NumDatapoints'+str(i+2)] = final_sub.NumDatapoints.iloc[i]
    combDF = combDF.append(new_comb_row, ignore_index=True)
combDF


# In[9]:


combDF.loc[combDF.Sec1 == 'US_Dollar_Index']


# <h1> Determine which sets of dates to try </h1>

# In[10]:


# [x] check that all secs in combDF exist in data_summary
secs_cols = ['Sec'+str(i) for i in range(1, max_sec_bundle+2)]
comb_secs = []
for s in secs_cols:
    comb_secs = np.append(comb_secs, combDF[s].values)
comb_secs = list(set(comb_secs))
comb_secs = [s for s in comb_secs if type(s)==str]
print('found '+str(len(comb_secs))+' unique securities in combDF')
print([s for s in comb_secs if s not in data_summaryDF.Name.values])
assert(np.all([s in data_summaryDF.Name.values for s in comb_secs]))


# In[11]:


# remove CRH_PLC
# raw_regressionDF = pd.read_csv(regression_results_fn)
# raw_regressionDF = raw_regressionDF.loc[(raw_regressionDF.Sec1 != 'CRH_PLC') & (raw_regressionDF.Sec2 != 'CRH_PLC')].reset_index(drop=True)
# raw_regressionDF.to_csv(regression_results_fn, index=False)S


# In[12]:


for i in range(1, max_sec_bundle+2):
    combDF['TotalStartDate'+str(i)] = np.nan
    combDF['TotalEndDate'+str(i)] = np.nan

for i in range(len(combDF)):
    #get total Start and End dates
    for j in range(1, max_sec_bundle+2):
        if type(combDF['Sec'+str(j)][i]) == str:
            combDF.loc[combDF['Sec'+str(j)] == combDF['Sec'+str(j)][i], 'TotalStartDate'+str(j)] = data_summaryDF.loc[data_summaryDF.Name == combDF['Sec'+str(j)][i], 'StartDate'].values[0]
            combDF.loc[combDF['Sec'+str(j)] == combDF['Sec'+str(j)][i], 'TotalEndDate'+str(j)] = data_summaryDF.loc[data_summaryDF.Name == combDF['Sec'+str(j)][i], 'EndDate'].values[0]


# In[13]:


combDF['LastStartDate'] = np.nan
combDF['FirstEndDate'] = np.nan
for i in combDF.index:
    comb_row = combDF.loc[i]
    start_dates = comb_row[[col for col in combDF.columns if 'TotalStartDate' in col]].values
    start_dates = [d for d in start_dates if type(d) == str]
    last_start_date = max(start_dates)
    end_dates = comb_row[[col for col in combDF.columns if 'TotalEndDate' in col]]
    end_dates = [d for d in end_dates if type(d) == str]
    first_end_date = min(end_dates)
    combDF.loc[i, 'LastStartDate'] = last_start_date
    combDF.loc[i, 'FirstEndDate'] = first_end_date
combDF


# In[14]:


# convert the date columns into actual dates
date_cols = [col for col in combDF.columns if 'Date' in col]
for col in date_cols:
    combDF[col] = pd.to_datetime(combDF[col]).dt.date


# In[15]:


min_training_days = 90
min_val_days = 60
train_pct = .66 #percent of train+val
max_val_date = '2019-09-01'
num_secondary_val_days = 60

combDF['TrainStartDate'], combDF['ValStartDate'], combDF['ValEndDate'] = np.nan, np.nan, np.nan

min_num_days_total = min_training_days + min_val_days + num_secondary_val_days
for i in combDF.index:
    num_days = (pd.to_datetime(combDF.loc[i, 'FirstEndDate']) - pd.to_datetime(combDF.loc[i, 'LastStartDate'])).days
    if num_days < min_num_days_total:
        print('\n'+'='*50+'num_days < min_num_days_total! Skipping row '+str(i)+':\n'+str(combDF.loc[i]))
    else:
        combDF.loc[i, 'TrainStartDate'] = combDF.loc[i, 'LastStartDate']
        combDF.loc[i, 'ValEndDate'] = combDF.loc[i, 'FirstEndDate'] - pd.Timedelta(days=num_secondary_val_days)
        combDF.loc[i, 'ValStartDate'] = combDF.loc[i, 'TrainStartDate'] + pd.Timedelta(days=int(num_days*train_pct))
combDF


# <h1> Determine types of models to try </h1>
# <br>
# This entails choosing both the feature set and the type of model.

# In[16]:


### Timescales (one-value and PDF?)
# - 1 min
# - 3 min
# - 5 min
# - 10 min

### Features
# - Time of day
# - Tick Imbalance
# - last N minutes
# - change on day
# - change on hour
# - change on 5 day
# - relative prices BOD
# - (optional: Dividends and Earnings)

### Models
# - Linear regression
#   - with and without regularization
# - Random Forest
# - XGBoost?


# <h1> Create folder system and summary sheet </h1>
#

# In[17]:


# -US_Dollar_Index
# --Config1
# ---Data (temporary)
# ---summary_sheet (human readable html?)
# ---- == Results Summary ==
# ---- == Configuration ==
# ---- == Model Details ==
# ---progress_notes (machine readable text)
# ---- == Configuration ==
# ---- == Model Iteration ==
# ---log?
# ---Plots
# ---Deep_Models
# ----model_name.hdf5


# In[56]:


def createConfigDir(comb_row, sec1_dir):
    """Creates the file structure AND the progress notes file and summary sheet file"""
    print('Creating config file structure for:\n'+sec1_dir+'\n')
    configs = [f for f in os.listdir(sec1_dir) if f[:6]=='Config']
    config_nums = [f.split('Config')[1] for f in configs if f.split('Config')[1].isdigit()]
    new_config_num = 1 if len(config_nums)==0 else max([int(i) for i in config_nums])+1
    config_dir = sec1_dir+'Config'+str(new_config_num)+'/'
    os.mkdir(config_dir)
    data_dir = config_dir+'Data/'
    os.mkdir(data_dir)
    deep_models_dir = config_dir+'Deep_Models/'
    os.mkdir(deep_models_dir)
    plots_dir = config_dir+'Plots/'
    os.mkdir(plots_dir)
    # [x] log
    log_fn = config_dir+'log.txt'
    with open(log_fn, 'a+') as log_file:
        log_file.write('='*50+'\nCreated log file at '+getChiTimeNow().strftime(format='%Y-%m-%d %H:%M:%S %Z%z')+' (Chicago time)\n'+'='*50+'\n')

    # [x] progress_notes.txt
    progress_notes_fn = config_dir+'progress_notes.txt'
    other_secs_cols = [col for col in comb_row.index if (col[:3]=='Sec' and col != 'Sec1')]
    other_secs = comb_row[other_secs_cols]
    other_secs = [sec for sec in other_secs if type(sec)==str]
    other_secs_idx = [i for i in range(2, len(other_secs)+2)]
    with open(progress_notes_fn, 'a+') as progress_notes_file:
        progress_notes_file.write('== Configuration ==\n')
        progress_notes_file.write('other_secs: '+','.join(other_secs)+'\n')
        progress_notes_file.write('TrainStartDate: '+comb_row.TrainStartDate.strftime(format='%Y-%m-%d')+'\n')
        progress_notes_file.write('ValStartDate: '+comb_row.ValStartDate.strftime(format='%Y-%m-%d')+'\n')
        progress_notes_file.write('ValEndDate: '+comb_row.ValEndDate.strftime(format='%Y-%m-%d')+'\n')
        progress_notes_file.write('== Model Iteration ==\n')
    # [x] summary_sheet.html
    col_stems = ['TotalStartDate', 'TotalEndDate', 'R2_', 'p_value', 'StartDate', 'EndDate', 'NumDatapoints']
    summaryDF = pd.DataFrame(index=['TotalStartDate', 'TotalEndDate', 'R2', 'p_value', 'StartDate', 'EndDate', 'NumDatapoints'])
    for sec_idx in other_secs_idx:
        summaryDF[comb_row['Sec'+str(sec_idx)]] = comb_row[[col+str(sec_idx) for col in col_stems]].values
    summary_sheet_fn = config_dir+'summary_sheet.html'
    with open(summary_sheet_fn, 'a+') as summary_sheet_file:
        summary_sheet_file.write('<h1>Results Summary</h1>')
        summary_sheet_file.write('<h1>Configuration</h1>')
        summary_sheet_file.write('<p>TrainStartDate: '+comb_row.TrainStartDate.strftime(format='%Y-%m-%d')+
                                 '<br>ValStartDate: '+comb_row.ValStartDate.strftime(format='%Y-%m-%d')+
                                 '<br>ValEndDate: '+comb_row.ValEndDate.strftime(format='%Y-%m-%d')+'</p>')
        # [x] save config table
        # [x] specify line_width in to_html below
        summary_sheet_file.write(summaryDF.to_html())#line_width=200)) #don't know why line_width isn't accepted here...
        summary_sheet_file.write('<h1>Model Details</h1>')
    print('Config file structure creation complete.\n')
    return(config_dir, data_dir, summary_sheet_fn, progress_notes_fn, log_fn, deep_models_dir)


# In[19]:


def locateConfigDir(comb_row, omni_dir):
    sec1_dir = omni_dir + comb_row.Sec1 +'/'
    if not os.path.exists(sec1_dir):
        print('Creating '+sec1_dir)
        os.mkdir(sec1_dir)
    config_dir = 'not set yet'
    configs = [f for f in os.listdir(sec1_dir) if f[:6]=='Config']
    # search for the appropriate configuration
    configs.sort()
    for config in configs:
        temp_notes_fn = sec1_dir+config+'/progress_notes.txt'
        if not os.path.exists(temp_notes_fn):
            print(temp_notes_fn+' doesnt exist!')
        else:
            temp_notes = 'not set yet'
            with open(temp_notes_fn,'r') as fh:
                temp_notes = fh.read()
            config_section = temp_notes.split('== Configuration ==\n')[1].split('\n== Model Iteration ==')[0]
            # [] TODO: Identify whether this is the correct configuration
            train_start_date = config_section.split('TrainStartDate: ')[1].split('\n')[0]
            val_start_date = config_section.split('ValStartDate: ')[1].split('\n')[0]
            val_end_date = config_section.split('ValEndDate: ')[1].split('\n')[0]
            other_secs = config_section.split('other_secs: ')[1].split('\n')[0].split(',')
            comb_row_other_secs_cols = [col for col in comb_row.index if (col[:3]=='Sec' and col != 'Sec1')]
            comb_row_other_secs = comb_row[comb_row_other_secs_cols]
            comb_row_other_secs = [sec for sec in comb_row_other_secs if type(sec)==str]
            if ((comb_row.TrainStartDate.strftime(format='%Y-%m-%d') == train_start_date) and
                (comb_row.ValStartDate.strftime(format='%Y-%m-%d') == val_start_date) and
                (comb_row.ValEndDate.strftime(format='%Y-%m-%d') == val_end_date) and
                (set(comb_row_other_secs) == set(other_secs))):
                config_dir = sec1_dir+config+'/'
                print('Existing matching config dir found: '+sec1_dir+config)
                return(config_dir, config_dir+'Data/', config_dir+'summary_sheet.html', config_dir+'progress_notes.txt', config_dir+'log.txt', config_dir+'Data/')
    return(createConfigDir(comb_row, sec1_dir))


# <h1> Iterate Learning </h1>

# <h3>Format/Process Data</h3>
def getExpectedPreprocessedYCols(y_min_incs):
    return_cols = ['Minute']
    for t in y_min_incs:
        return_cols.append('y_B'+str(t))
        return_cols.append('y_A'+str(t))
    return(return_cols)

def getExpectedPreprocessedXCols(num_secs, day_chg_incs, minute_incs):
    # there is probably a more elegant way to do this
    return_cols = ['Minute']
    # outer loop is always SEC for [DAY], [MINUTE] is always the outer loop for [SEC]. ...I think
    # [SEC1] means starts at sec1, [SEC2] means starts at sec2
    chunk1 = ['O_B[SEC1]', 'O_A[SEC1]', 'H_B[SEC1]', 'H_A[SEC1]', 'L_B[SEC1]', 'L_A[SEC1]', 'C_B[SEC1]', 'C_A[SEC1]',
              'Count[SEC1]', 'B_TickImb[SEC1]', 'A_TickImb[SEC1]', 'M_TickImb[SEC1]']
    chunk2 = ['Sec[SEC1]_Open_B', 'Sec[SEC1]_Open_A',
              'Sec[SEC1]_Open_B_chg[DAY]', 'Sec[SEC1]_Open_A_chg[DAY]']
    chunk3 = ['Sec[SEC2]_Open_B_Quotient', 'Sec[SEC2]_Open_A_Quotient']
    chunk4 = ['O_B[SEC1]_diff[MINUTE]', 'O_A[SEC1]_diff[MINUTE]', 'H_B[SEC1]_diff[MINUTE]', 'H_A[SEC1]_diff[MINUTE]', 'L_B[SEC1]_diff[MINUTE]', 'L_A[SEC1]_diff[MINUTE]', 'C_B[SEC1]_diff[MINUTE]', 'C_A[SEC1]_diff[MINUTE]']
    chunk5 = ['Count[SEC1]_sum[MINUTE]', 'B_TickImb[SEC1]_sum[MINUTE]', 'A_TickImb[SEC1]_sum[MINUTE]', 'M_TickImb[SEC1]_sum[MINUTE]']
    chunks = [chunk1, chunk2, chunk3, chunk4, chunk5]
    for chunk in chunks:
        out_chunk = []
        for t_col in chunk:
            if ('[SEC1]' in t_col) and ('[DAY]' in t_col):
                for s in range(1, (num_secs+1)):
                    for d in day_chg_incs:
                        out_chunk.append(t_col.replace('[SEC1]', str(s)).replace('[DAY]', str(d)))
            elif ('[SEC1]' in t_col) and ('[MINUTE]' in t_col):
                for s in range(1, (num_secs+1)):
                    for m in minute_incs:
                        out_chunk.append(t_col.replace('[SEC1]', str(s)).replace('[MINUTE]', str(m)))
            elif ('[SEC2]' in t_col) and ('[DAY]' in t_col):
                for s in range(2, (num_secs+1)):
                    for d in day_chg_incs:
                        out_chunk.append(t_col.replace('[SEC2]', str(s)).replace('[DAY]', str(d)))
            elif ('[SEC2]' in t_col) and ('[MINUTE]' in t_col):
                for s in range(2, (num_secs+1)):
                    for m in minute_incs:
                        out_chunk.append(t_col.replace('[SEC2]', str(s)).replace('[MINUTE]', str(m)))
            elif '[SEC1]' in t_col:
                for s in range(1, (num_secs+1)):
                    out_chunk.append(t_col.replace('[SEC1]', str(s)))
            elif '[SEC2]' in t_col:
                for s in range(2, (num_secs+1)):
                    out_chunk.append(t_col.replace('[SEC2]', str(s)))
            else:
                raise ValueError('Unrecognized template col: '+t_col)
        return_cols += out_chunk
    return(return_cols)

# In[20]:


### Features
# [x] - Time of day
# [NA] - Tick Imbalance
# [x] - last [5, 15, 30, 60] minutes
# [NA] - change on day
# [x] - change on [1,3,5] day
# [x] - relative prices BOD

### Timescales (one-value and PDF?)
# - 1 min
# - 3 min
# - 5 min
# - 10 min


def processData(all_minutesDF, dailyDF, sec_guideDF, val_start_date, config_dir, saveProcessed=False):
    saveProcessed = True
    day_chg_incs = [1, 3, 5]
    minute_incs = [5, 15, 30, 60]
    y_min_incs = [1, 3, 5, 10]
    num_secs = len(sec_guideDF)
    print('In processData. Checking for preprocessedData...') #might want to move this upstream depending on speed
    #look through preprocessed data folder
    preprocessed_data_dir = config_dir + 'Preprocessed_Data/'
    if os.path.exists(preprocessed_data_dir):
        if len(os.listdir(preprocessed_data_dir)) >= 4:
            print('preprocessedData found. Checking columns and returning')
            # need to check the columns
            y_cols_expected = getExpectedPreprocessedYCols(y_min_incs)
            y_cols_actual =  pd.read_csv(preprocessed_data_dir+'train_minutesDF_y.csv', nrows=0).columns.tolist()
            assert y_cols_expected == y_cols_actual
            # check that the X_col sets are the same. Assume the ordering is correct.
            X_cols_set_expected = set(getExpectedPreprocessedXCols(num_secs, day_chg_incs, minute_incs))
            X_cols_set_actual =  set(pd.read_csv(preprocessed_data_dir+'train_minutesDF_X.csv', nrows=0).columns.tolist())
            assert X_cols_set_expected == X_cols_set_actual, str(X_cols_set_expected.difference(X_cols_set_actual))+'\n'+str(X_cols_set_actual.difference(X_cols_set_expected))
            print('loading train_minutesDF_X...')
            train_minutesDF_X = pd.read_csv(preprocessed_data_dir+'train_minutesDF_X.csv', parse_dates=['Minute'])
            print('loading train_minutesDF_y...')
            train_minutesDF_y = pd.read_csv(preprocessed_data_dir+'train_minutesDF_y.csv', parse_dates=['Minute'])
            print('loading val_minutesDF_X...')
            val_minutesDF_X = pd.read_csv(preprocessed_data_dir+'val_minutesDF_X.csv', parse_dates=['Minute'])
            print('loading val_minutesDF_y...')
            val_minutesDF_y = pd.read_csv(preprocessed_data_dir+'val_minutesDF_y.csv', parse_dates=['Minute'])
            return(train_minutesDF_X, train_minutesDF_y, val_minutesDF_X, val_minutesDF_y)
    print('Preprocessed data not found. Processing data from all_minutesDF...')
    assert(np.all(all_minutesDF.index == range(len(all_minutesDF))))
    # [x] - Time of day
    all_minutesDF.TimeNumeric = all_minutesDF.Minute.dt.hour*60+all_minutesDF.Minute.dt.minute
    # [x] fill in the NAs
    print('filling in NAs...')
    ffill_col_stems = ['C_B', 'C_A']
    take_filled_close_col_stems = ['O_B', 'O_A', 'H_B', 'H_A', 'L_B', 'L_A']
    zerofill_col_stems = ['Count', 'B_TickImb', 'A_TickImb', 'M_TickImb']
    for sec_num in range(1, num_secs+1):
        ffill_cols = [s+str(sec_num) for s in ffill_col_stems]
        take_filled_close_cols = [s+str(sec_num) for s in take_filled_close_col_stems]
        zerofill_cols = [s+str(sec_num) for s in zerofill_col_stems]
        all_minutesDF.loc[:,ffill_cols] = all_minutesDF.loc[:,ffill_cols].ffill()
        all_minutesDF.loc[:,zerofill_cols] = all_minutesDF.loc[:,zerofill_cols].fillna(0)
        B_take_filled_close_cols = [col for col in take_filled_close_cols if '_B' in col]
        A_take_filled_close_cols = [col for col in take_filled_close_cols if '_A' in col]
        # use fillna below to achieve successful broadcasting of C_B# and C_A# columns
        all_minutesDF.loc[all_minutesDF[B_take_filled_close_cols[0]].isna(), B_take_filled_close_cols] = \
            all_minutesDF.loc[all_minutesDF[B_take_filled_close_cols[0]].isna(), B_take_filled_close_cols].fillna(0).add(
                all_minutesDF.loc[all_minutesDF[B_take_filled_close_cols[0]].isna(), 'C_B'+str(sec_num)], axis=0)
        all_minutesDF.loc[all_minutesDF[A_take_filled_close_cols[0]].isna(), A_take_filled_close_cols] = \
            all_minutesDF.loc[all_minutesDF[A_take_filled_close_cols[0]].isna(), A_take_filled_close_cols].fillna(0).add(
                all_minutesDF.loc[all_minutesDF[A_take_filled_close_cols[0]].isna(), 'C_A'+str(sec_num)], axis=0)
    # [x] - change on [1,3,5] day
    print('calculating change on '+str(day_chg_incs)+' days...')
    dailyDF['OpenDT'] = pd.to_datetime(dailyDF.Date.astype(str)+' '+dailyDF.Open, format='%Y-%m-%d %H:%M')
    dailyDF['CloseDT'] = pd.to_datetime(dailyDF.Date.astype(str)+' '+dailyDF.Close, format='%Y-%m-%d %H:%M')
    for sec_num in range(1, num_secs+1):
        dailyDF['Sec'+str(sec_num)+'_Open_B'], dailyDF['Sec'+str(sec_num)+'_Open_A'] = np.nan, np.nan
        dailyDF[['Sec'+str(sec_num)+'_Open_B', 'Sec'+str(sec_num)+'_Open_A']] = \
            pd.merge(dailyDF[['OpenDT']], all_minutesDF[['Minute', 'C_B'+str(sec_num), 'C_A'+str(sec_num)]],
                     left_on=['OpenDT'], right_on=['Minute'], how='left')[['C_B'+str(sec_num), 'C_A'+str(sec_num)]]
        for day_inc in day_chg_incs:
            # today's open is the reference point for change on X day
            dailyDF['Sec'+str(sec_num)+'_Open_B_chg'+str(day_inc)] = \
                dailyDF['Sec'+str(sec_num)+'_Open_B'].shift(day_inc) - dailyDF['Sec'+str(sec_num)+'_Open_B']
            dailyDF['Sec'+str(sec_num)+'_Open_A_chg'+str(day_inc)] = \
                dailyDF['Sec'+str(sec_num)+'_Open_A'].shift(day_inc) - dailyDF['Sec'+str(sec_num)+'_Open_A']
    # [x] - relative prices BOD (quotient)
    for sec_num in range(2, num_secs+1):
        dailyDF['Sec'+str(sec_num)+'_Open_B_Quotient'] = dailyDF['Sec'+str(sec_num)+'_Open_B']/dailyDF['Sec1_Open_B']
        dailyDF['Sec'+str(sec_num)+'_Open_A_Quotient'] = dailyDF['Sec'+str(sec_num)+'_Open_A']/dailyDF['Sec1_Open_A']
    new_cols = [col for col in dailyDF.columns if col not in ['Date', 'Open', 'Close', 'OpenDT', 'CloseDT', 'Unnamed: 0']]
    for col in new_cols: all_minutesDF[col] = np.nan
    all_minutesDF[new_cols] = pd.merge(all_minutesDF[['Date']], dailyDF[['Date']+new_cols],
                                       on='Date', how='left')[new_cols]
    # [x] make all prices relative to BOD
    print('making all prices relative to BOD...')
    col_stems_to_make_relative = ['O_B', 'O_A', 'H_B', 'H_A', 'L_B', 'L_A', 'C_B', 'C_A']
    for sec_num in range(1, num_secs+1):
        B_cols_to_make_relative = [s+str(sec_num) for s in col_stems_to_make_relative if '_B' in s]
        A_cols_to_make_relative = [s+str(sec_num) for s in col_stems_to_make_relative if '_A' in s]
        all_minutesDF[B_cols_to_make_relative] = all_minutesDF[B_cols_to_make_relative].subtract(
            pd.merge(all_minutesDF[['Date']], dailyDF[['Date', 'Sec'+str(sec_num)+'_Open_B']],
                     on='Date', how='left')['Sec'+str(sec_num)+'_Open_B'], axis=0)
        all_minutesDF[A_cols_to_make_relative] = all_minutesDF[A_cols_to_make_relative].subtract(
            pd.merge(all_minutesDF[['Date']], dailyDF[['Date', 'Sec'+str(sec_num)+'_Open_A']],
                     on='Date', how='left')['Sec'+str(sec_num)+'_Open_A'], axis=0)
    # [x] - last [5, 15, 30, 60] minutes
    print('getting data for last '+str(minute_incs)+' minutes')
    col_stems_to_add = ['Count', 'B_TickImb', 'A_TickImb', 'M_TickImb']
    col_stems_to_diff = ['O_B', 'O_A', 'H_B', 'H_A', 'L_B', 'L_A', 'C_B', 'C_A']
    cols_to_add, cols_to_diff = [], []
    for sec_num in range(1, num_secs+1):
        cols_to_add += [s+str(sec_num) for s in col_stems_to_add]
        cols_to_diff += [s+str(sec_num) for s in col_stems_to_diff]
    new_cols_add, new_cols_diff = [], []
    for min_inc in minute_incs:
        new_cols_add += [col+'_sum'+str(min_inc) for col in cols_to_add]
        new_cols_diff += [col+'_diff'+str(min_inc) for col in cols_to_diff]
    # diff the diff columns, then fill in the first min_inc rows of each date with the BOD value
    print('creating minute diff cols...')
    for min_inc in minute_incs:
        all_minutesDF[[col+'_diff'+str(min_inc) for col in cols_to_diff]] = all_minutesDF[cols_to_diff].shift(min_inc)
        for date in all_minutesDF.Date.unique():
            date_subDF = all_minutesDF.loc[all_minutesDF.Date == date]
            repl_subDF = date_subDF.iloc[:min_inc]
            all_minutesDF.loc[repl_subDF.index, [col+'_diff'+str(min_inc) for col in cols_to_diff]] = \
                repl_subDF[cols_to_diff].iloc[0].values
    # add the add columns, making sure not to go past the current date
    print('creating minute add cols...')
    for min_inc in minute_incs:
        for col in cols_to_add: all_minutesDF[col+'_sum'+str(min_inc)] = np.nan
    for date in all_minutesDF.Date.unique():
        date_subDF = all_minutesDF.loc[all_minutesDF.Date == date]
        for min_inc in minute_incs:
            all_minutesDF.loc[date_subDF.index, [col+'_sum'+str(min_inc) for col in cols_to_add]] = \
                date_subDF[cols_to_add].rolling(min_inc, min_periods=1).sum().values
    # [x] create y columns
    print('creating y columns for future '+str(y_min_incs)+' minutes...')
    for y_min_inc in y_min_incs:
        all_minutesDF['y_B'+str(y_min_inc)] = np.nan
        all_minutesDF['y_A'+str(y_min_inc)] = np.nan
        for date in all_minutesDF.Date.unique():
            date_subDF = all_minutesDF.loc[all_minutesDF.Date == date]
            all_minutesDF.loc[date_subDF.index, 'y_B'+str(y_min_inc)] = date_subDF.C_B1.shift(-y_min_inc)
            all_minutesDF.loc[date_subDF.index, 'y_A'+str(y_min_inc)] = date_subDF.C_A1.shift(-y_min_inc)
            # fill NAs at the end of the day with the latest available data that day
            date_subDF = all_minutesDF.loc[all_minutesDF.Date == date] # need this for the line below
            na_subDF = date_subDF.loc[date_subDF['y_B'+str(y_min_inc)].isna()]
            all_minutesDF.loc[na_subDF.index, ['y_B'+str(y_min_inc), 'y_A'+str(y_min_inc)]] = na_subDF[['C_B1', 'C_A1']].iloc[-1].values
        # make y columns relative to current close
        all_minutesDF['y_B'+str(y_min_inc)] = all_minutesDF['y_B'+str(y_min_inc)] - all_minutesDF.C_B1
        all_minutesDF['y_A'+str(y_min_inc)] = all_minutesDF['y_A'+str(y_min_inc)] - all_minutesDF.C_A1
    # [x] subset data and return
    print('subsetting data and returning...')
    days_to_delete = max(day_chg_incs)
    train_start_date = all_minutesDF.Date.unique()[days_to_delete]
    train_end_date = val_start_date
    # add val_start_date into all_dates, sort, find the index, then remove the addition
    all_dates = list(all_minutesDF.Date.unique())
    all_dates.append(val_start_date)
    all_dates.sort()
    val_start_date_idx = all_dates.index(val_start_date)
    all_dates.pop(val_start_date_idx)
    assert(val_start_date_idx > days_to_delete)
    val_start_date = all_minutesDF.Date.unique()[val_start_date_idx+days_to_delete]
    train_minutesDF = all_minutesDF.loc[(all_minutesDF.Date >= train_start_date) & (all_minutesDF.Date <= train_end_date)].reset_index(drop=True)
    val_minutesDF = all_minutesDF.loc[(all_minutesDF.Date >= val_start_date)].reset_index(drop=True)
    y_cols = ['y_B'+str(y_min_inc) for y_min_inc in y_min_incs] + ['y_A'+str(y_min_inc) for y_min_inc in y_min_incs]
    y_cols.sort(key=lambda x: int(x[3:])) #puts them in y_min_inc order
    unused_cols = ['Unnamed: 0', 'Product', 'Date', 'Minute', 'First', 'Last']
    X_cols = [col for col in all_minutesDF.columns if (col not in y_cols+unused_cols)]
    # UNCOMMENT BELOW!!
    assert(np.all(~train_minutesDF[X_cols].isna()))
    assert(np.all(~val_minutesDF[X_cols].isna()))
    assert(np.all(~train_minutesDF[y_cols].isna()))
    assert(np.all(~val_minutesDF[y_cols].isna()))
    X_cols, y_cols = ['Minute']+list(X_cols), ['Minute']+list(y_cols)# We still need the minutes
    if saveProcessed:
        print('Saving processed data')
        if not os.path.exists(preprocessed_data_dir): os.mkdir(preprocessed_data_dir)
        train_minutesDF[X_cols].to_csv(preprocessed_data_dir+'train_minutesDF_X.csv', index=False)
        train_minutesDF[y_cols].to_csv(preprocessed_data_dir+'train_minutesDF_y.csv', index=False)
        val_minutesDF[X_cols].to_csv(preprocessed_data_dir+'val_minutesDF_X.csv', index=False)
        val_minutesDF[y_cols].to_csv(preprocessed_data_dir+'val_minutesDF_y.csv', index=False)
    print('Returning train with '+'{:,}'.format(len(train_minutesDF))+' rows and val with '+'{:,}'.format(len(val_minutesDF))+' rows.')
    return(train_minutesDF[X_cols], train_minutesDF[y_cols], val_minutesDF[X_cols], val_minutesDF[y_cols])

# In[58]:


def loadData(comb_row, omni_dir, config_dir, minute_dir):
    data_dir = config_dir+'Data/'
    # [x] check if we already have data loaded
    daily_fn = data_dir+'daily_summary.csv'
    all_minutes_fn = data_dir+'all_minutes.csv'
    sec_guide_fn = data_dir+'sec_guide.csv'
    all_minutesDF = 'not set yet'
    if os.path.exists(daily_fn) and os.path.exists(all_minutes_fn) and os.path.exists(sec_guide_fn):
        print('Data found! Loading data...')
        # [x] check the order of other_secs
        sec_guideDF = pd.read_csv(sec_guide_fn)
        assert(comb_row.Sec1 == sec_guideDF.iloc[0].Sec)
        other_secs_comb_row = comb_row[[col for col in comb_row.index if (col[:3]=='Sec' and col!='Sec1')]].values
        other_secs_sec_guide = sec_guideDF.iloc[1:].Sec.values
        assert(np.all(other_secs_comb_row == other_secs_sec_guide))
        # [x] check the dates
        dailyDF = pd.read_csv(daily_fn, parse_dates=['Date'])
        dailyDF.Date = dailyDF.Date.dt.date
        first_date = dailyDF.Date.iloc[0]
        last_date = dailyDF.Date.iloc[-1]
        assert(abs((pd.to_datetime(first_date) - pd.to_datetime(comb_row.TrainStartDate)).days) < 1) # may want to relax these
        assert(abs((pd.to_datetime(last_date) - pd.to_datetime(comb_row.ValEndDate)).days) < 1)
        all_minutesDF = pd.read_csv(all_minutes_fn, parse_dates=['Date', 'Minute'])
        all_minutesDF.Date = all_minutesDF.Date.dt.date
        print('Data load complete.')
    else:
        other_secs = comb_row[[col for col in comb_row.index if (col[:3]=='Sec' and col!='Sec1')]].values
        print('No pre-loaded data found. Loading data for '+comb_row.Sec1+' and '+','.join(other_secs))
        sec1_minuteDF = pd.read_csv(minute_dir+comb_row.Sec1+'.csv', parse_dates=['Date', 'Minute'])
        sec1_minuteDF.Date = sec1_minuteDF.Date.dt.date
        sec1_minuteDF = sec1_minuteDF.loc[(sec1_minuteDF.Date >= comb_row.TrainStartDate) & (sec1_minuteDF.Date <= comb_row.ValEndDate)].reset_index(drop=True)

        print('Loading minuteDF for '+comb_row.Sec2)

        other_secs_minuteDF = pd.read_csv(minute_dir+comb_row.Sec2+'.csv', parse_dates=['Date', 'Minute'])[['Product', 'Date', 'Minute', 'O_B', 'O_A', 'H_B', 'H_A', 'L_B', 'L_A', 'C_B', 'C_A', 'Count', 'B_TickImb', 'A_TickImb', 'M_TickImb']]
        for sec in other_secs[1:]:
            other_secs_minuteDF = other_secs_minuteDF.append(pd.read_csv(minute_dir+sec+'.csv', parse_dates=['Date', 'Minute'])[['Product', 'Date', 'Minute', 'O_B', 'O_A', 'H_B', 'H_A', 'L_B', 'L_A', 'C_B', 'C_A', 'Count', 'B_TickImb', 'A_TickImb', 'M_TickImb']], ignore_index=True)
        other_secs_minuteDF.Date = other_secs_minuteDF.Date.dt.date
        other_secs_minuteDF = other_secs_minuteDF.loc[(other_secs_minuteDF.Date >= comb_row.TrainStartDate) & (other_secs_minuteDF.Date <= comb_row.ValEndDate)].reset_index(drop=True)
        print('other_secs_minuteDF has '+str(len(other_secs_minuteDF))+' rows.')
        print('sec1_minuteDF has '+str(len(sec1_minuteDF))+' rows.')
        print("pd.read_csvs complete. Subsetting dates...")
        # [x] subset for dates
        dates_in_common = set(sec1_minuteDF.Date.unique())
        for sec in other_secs:
            dates_in_common = dates_in_common.intersection(set(other_secs_minuteDF.loc[other_secs_minuteDF.Product == sec].Date.unique()))
        sec1_dates_to_remove = set(sec1_minuteDF.Date.unique()).difference(dates_in_common)
        print(str(len(dates_in_common))+' dates_in_common')
        if len(sec1_dates_to_remove) > 0:
            print('- removing '+str(len(sec1_dates_to_remove))+' dates from '+comb_row.Sec1)
            sec1_minuteDF = sec1_minuteDF.loc[sec1_minuteDF.Date.isin(sec1_dates_to_remove)].reset_index(drop=True)
        for sec in other_secs:
            sec_dates_to_remove = set(other_secs_minuteDF.loc[other_secs_minuteDF.Product==sec].Date.unique()).difference(dates_in_common)
            if len(sec_dates_to_remove) > 0:
                print('- removing '+str(len(sec_dates_to_remove))+' dates from '+sec)
        other_secs_minuteDF = other_secs_minuteDF.loc[other_secs_minuteDF.Date.isin(dates_in_common)].reset_index(drop=True)
        print('other_secs_minuteDF has '+str(len(other_secs_minuteDF))+' rows.')
        print('sec1_minuteDF has '+str(len(sec1_minuteDF))+' rows.')
        print("Date subset complete. Determining each day's Open/Closes...")
        # [x] determine each day's open and close
        dailyDF = pd.DataFrame(columns=['Date', 'Open', 'Close'])
        dailyDF['Date'] = sec1_minuteDF.Date.unique()
        for i in tqdm.tqdm(range(len(dailyDF))):
            date = dailyDF.loc[i].Date
            lastOpen = sec1_minuteDF.loc[sec1_minuteDF.Date==date].Minute.min()
            firstClose = sec1_minuteDF.loc[sec1_minuteDF.Date==date].Minute.max()
            other_sec_date_subDF = other_secs_minuteDF.loc[other_secs_minuteDF.Date == date]
            for sec in other_secs:
                lastOpen = max(lastOpen, other_sec_date_subDF.loc[other_sec_date_subDF.Product==sec].Minute.min())
                firstClose = min(firstClose, other_sec_date_subDF.loc[other_sec_date_subDF.Product==sec].Minute.max())
            dailyDF.loc[i, 'Open'] = lastOpen
            dailyDF.loc[i, 'Close'] = firstClose
        dailyDF.Open = dailyDF.Open.dt.strftime(date_format='%H:%M')
        dailyDF.Close = dailyDF.Close.dt.strftime(date_format='%H:%M')
        dailyDF.to_csv(data_dir+'daily_summary.csv', index=False)
        print("Each day's Open/Closes determination complete. Creating all_minutesDF...")
        # [x] create all_minutesDF
        all_minutesDF = pd.DataFrame(columns=['Date', 'Minute'])
        # enumerate minutes
        for i in range(len(dailyDF)):
            open_dt = pd.to_datetime(dailyDF.loc[i].Date.strftime(format='%Y-%m-%d')+' '+dailyDF.loc[i].Open, format='%Y-%m-%d %H:%M')
            close_dt = pd.to_datetime(dailyDF.loc[i].Date.strftime(format='%Y-%m-%d')+' '+dailyDF.loc[i].Close, format='%Y-%m-%d %H:%M')
            minute_range = pd.date_range(start=open_dt, end=close_dt, freq='T')
            day_minutesDF = pd.DataFrame({'Date': minute_range.date, 'Minute': minute_range.values})
            all_minutesDF = all_minutesDF.append(day_minutesDF, ignore_index=True)
        #populate minute data
        col_stems = ['O_B', 'O_A', 'H_B', 'H_A', 'L_B', 'L_A', 'C_B', 'C_A', 'Count', 'B_TickImb', 'A_TickImb', 'M_TickImb']
        first_minute_populate_stems = ['O_B', 'O_A', 'H_B', 'H_A', 'L_B', 'L_A', 'C_B', 'C_A']
        for sec_num in range(1, len(other_secs)+2):
            sec_cols = [col_stem+str(sec_num) for col_stem in col_stems]
            for sec_col in sec_cols: all_minutesDF[sec_col] = np.nan
        all_minutesDF[[c+'1' for c in col_stems]] = pd.merge(all_minutesDF[['Minute']], sec1_minuteDF[['Minute']+col_stems], on='Minute', how='left')[col_stems]
        print('Merging into all_minutesDF...')
        for sec_num in range(2, len(other_secs)+2):
            other_sec = other_secs[sec_num-2]
            all_minutesDF[[c+str(sec_num) for c in col_stems]] = pd.merge(all_minutesDF[['Minute']], other_secs_minuteDF[['Minute']+col_stems].loc[other_secs_minuteDF.Product==other_sec], on='Minute', how='left')[col_stems]
        print('Getting the first datapoint of each day...')
        #get first datapoint of each day
        for i in tqdm.tqdm(dailyDF.index):
            date = dailyDF.loc[i].Date
            open_dt = pd.to_datetime(dailyDF.loc[i].Date.strftime(format='%Y-%m-%d')+' '+dailyDF.loc[i].Open, format='%Y-%m-%d %H:%M')
            sec1_last_row = sec1_minuteDF.loc[(sec1_minuteDF.Date == date) & (sec1_minuteDF.Minute <= open_dt)].iloc[-1]
            if sec1_last_row.Minute < open_dt:
                if (open_dt - sec1_last_row.Minute).seconds/60 > 20:
                    raise ValueError('Too much time has elapsed. '+comb_row.Sec1+' open quote is stale at '+open_dt.strftime(format='%Y-%m-%d %H:%M')+' by '+str((open_dt - sec1_last_row.Minute).seconds/60)+' minutes.')
                else:
                    all_minutesDF.loc[all_minutesDF.Minute == open_dt, [c+'1' for c in col_stems]] = 0
                    all_minutesDF.loc[all_minutesDF.Minute == open_dt, [c+'1' for c in first_minute_populate_stems]] = sec1_last_row[first_minute_populate_stems]
            other_secs_subDF = other_secs_minuteDF.loc[(other_secs_minuteDF.Date == date) & (other_secs_minuteDF.Minute <= open_dt)]
            for sec_num in range(2, len(other_secs)+2):
                other_sec = other_secs[sec_num-2]
                other_sec_last_row = other_secs_subDF.loc[other_secs_subDF.Product==other_sec].iloc[-1]
                if other_sec_last_row.Minute < open_dt:
                    if (open_dt - other_sec_last_row.Minute).seconds/60 > 20:
                        raise ValueError("Too much time has elapsed. "+other_sec+" open quote is stale at "+date.strftime(open_dt='%Y-%m-%d %H:%M')+' by '+str((open_dt - other_sec_last_row.Minute).seconds/60)+' minutes.')
                    else:
                        all_minutesDF.loc[all_minutesDF.Minute == open_dt, [c+str(sec_num) for c in col_stems]] = 0
                        all_minutesDF.loc[all_minutesDF.Minute == open_dt, [c+str(sec_num) for c in first_minute_populate_stems]] = other_sec_last_row[first_minute_populate_stems]
        print('Saving all_minutesDF...')
        all_minutesDF.to_csv(data_dir+'all_minutes.csv', index=False)
        print('Save complete.')
        sec_guideDF = pd.DataFrame({'Sec': [comb_row.Sec1]+list(other_secs)})
        sec_guideDF.to_csv(data_dir+'sec_guide.csv', index=False)
    return(processData(all_minutesDF, dailyDF, sec_guideDF, comb_row.ValStartDate, config_dir))


# <h3>Iterate on models</h3>

# In[22]:


model_order = ['ZM', 'simple_LM', 'ridge_LM', 'norm_ridge_LM', 'lasso_LM', 'norm_lasso_LM', 'random_forest', 'XGBoost']
model_order_dict = {}
for i,mod in enumerate(model_order): model_order_dict[mod]=i


# In[23]:


### Models
# - Linear regression
#   - with and without regularization
#   - with and without normalization
# - Random Forest
# - XGBoost?

def getModelToDoDict(config_dir, trainY, model_names=model_order):
    y_min_incs = [int(i.split('y_B')[1]) for i in trainY.columns if i[:3]=='y_B']
    to_do_dict = {};
    for y_min_inc in y_min_incs: to_do_dict[y_min_inc] = model_names[:]
    done_dict = {}
    print('Searching for current model progress')
    summary_sheet_fn = config_dir+'summary_sheet.html'
    summary_sheet_html = 'not set yet'
    with open(summary_sheet_fn,'r') as fh:
        summary_sheet_html = fh.read()
    # [x] rewrite with soup
    summary_soup = Soup(summary_sheet_html)
    all_models_soup = summary_soup.select('#all_models_table')
    if len(all_models_soup) == 0:
        print('No all_models_soup. Assuming no models have been done.')
    else:
        all_models_table = all_models_soup[0]
        all_modelsDF = pd.read_html(str(all_models_table))[0] #don't know why [0] is necessary
        for y in all_modelsDF.y.unique():
            done_dict[y] = list(all_modelsDF.loc[all_modelsDF.y == y].Name.values)
            done_dict[y] = [m for m in done_dict[y] if m[0]!='['] # does not get Keras models this way

    for done_y_min_inc_key in done_dict.keys():
        for model_done in done_dict[done_y_min_inc_key]:
            if model_done in to_do_dict[done_y_min_inc_key]:
                to_do_dict[done_y_min_inc_key].remove(model_done)
            else:
                print('\nWeird! For y_minute='+str(y_min_inc)+' we have done an extra model: '+model_done+'\n')
    if len(done_dict.keys()) == 0:
        print('\nNo models found. Models to go: '+str(to_do_dict)+'...')
    else:
        print('Some models already done: '+str(done_dict)+'\n\nModels to go: '+str(to_do_dict))
    return(to_do_dict)


# In[24]:


def formatYData(trainY, valY, y_min_inc):
    trainY['y'+str(y_min_inc)] = (trainY['y_B'+str(y_min_inc)] + trainY['y_A'+str(y_min_inc)])/2.0
    valY['y'+str(y_min_inc)] = (valY['y_B'+str(y_min_inc)] + valY['y_A'+str(y_min_inc)])/2.0
    return(trainY[['Minute', 'y'+str(y_min_inc)]], valY[['Minute', 'y'+str(y_min_inc)]])


# In[25]:


def getRegressionSummary(model, X, y):
    """
    https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
    """
    params = np.append(model.intercept_,model.coef_)
    predictions = model.predict(X)

#     newX = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X))
#     newX = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X.reset_index(drop=True)))
#     MSE = (sum((y-predictions)**2))/(len(newX)-len(newX.columns))

    # Note if you don't want to use a DataFrame replace the two lines above with
    newX = np.append(np.ones((len(X),1)), X, axis=1)
    MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

    var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params/ sd_b
    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]
    sd_b = np.round(sd_b,3)
    ts_b = np.round(ts_b,3)
    p_values = np.round(p_values,3)
    params = np.round(params,4)
    myDF3 = pd.DataFrame()
    myDF3['Var'] = ['Const']+list(X.columns) #my modification to add the variable names
    myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["PValue"] = [params,sd_b,ts_b,p_values]
#     print(myDF3)
    return(myDF3)


# In[26]:


def getSKModelResults(model, trainX, trainY_min_inc, valX, valY_min_inc, train_minutes, val_minutes, x_scaler=None, y_scaler=None):
    try:
        summaryDF = getRegressionSummary(model, trainX, trainY_min_inc)
    except:
        print('Error getting summaryDF')
        summaryDF = None
    train_preds = model.predict(trainX)
    val_preds = model.predict(valX)
    if x_scaler is not None:
        trainX, trainY_min_inc, valX, valY_min_inc = unnormalizeData(trainX, trainY_min_inc, valX, valY_min_inc, x_scaler, y_scaler)
        train_preds = unnormalizeIndividualData(train_preds, y_scaler)
        val_preds = unnormalizeIndividualData(val_preds, y_scaler)
    train_R2 = r2_score(trainY_min_inc, train_preds)
    val_R2 = r2_score(valY_min_inc, val_preds)
    train_resid = train_preds - trainY_min_inc
    val_resid = val_preds - valY_min_inc
    train_mse = np.sum(np.square(train_resid))/len(train_preds)
    val_mse = np.sum(np.square(val_resid))/len(val_preds)
    train_seDF = pd.DataFrame({'Minute': train_minutes.values, 'SE': np.square(train_resid)})
    train_seDF['Date'] = train_seDF.Minute.dt.date
    train_mse_by_dateDF = train_seDF.groupby('Date').agg({'SE': 'mean'})
    val_seDF = pd.DataFrame({'Minute': val_minutes.values, 'SE': np.square(val_resid)})
    val_seDF['Date'] = val_seDF.Minute.dt.date
    val_mse_by_dateDF = val_seDF.groupby('Date').agg({'SE': 'mean'})
    train_mse_by_dateDF.rename(columns={'SE': 'MSE'}, inplace=True); val_mse_by_dateDF.rename(columns={'SE': 'MSE'}, inplace=True)
    return(train_R2, val_R2, train_mse, train_mse_by_dateDF, val_mse, val_mse_by_dateDF, summaryDF)


# In[27]:


def prepDataForSKModel(trainX, trainY_min_inc, valX, valY_min_inc):
    train_minutes, val_minutes = trainY_min_inc.Minute, valY_min_inc.Minute
    trainX = trainX.drop(columns=['Minute']).astype(float)
    trainY_min_inc = trainY_min_inc.drop(columns=['Minute']).values.astype(float)
    trainY_min_inc = trainY_min_inc.reshape(len(trainY_min_inc))
    valX = valX.drop(columns=['Minute']).astype(float)
    valY_min_inc = valY_min_inc.drop(columns=['Minute']).values.astype(float)
    valY_min_inc = valY_min_inc.reshape(len(valY_min_inc))
    return(train_minutes, trainX, trainY_min_inc, val_minutes, valX, valY_min_inc)


# In[28]:


def zeroModel(trainX, trainY_min_inc, valX, valY_min_inc):
    print('in zeroModel')
    train_minutes, trainX, trainY_min_inc, val_minutes, valX, valY_min_inc =         prepDataForSKModel(trainX, trainY_min_inc, valX, valY_min_inc)
    train_R2 = r2_score(trainY_min_inc, np.zeros_like(trainY_min_inc))
    train_seDF = pd.DataFrame({'Minute': train_minutes.values, 'SE': np.square(trainY_min_inc)})
    train_seDF['Date'] = train_seDF.Minute.dt.date
    train_mse_by_dateDF = train_seDF.groupby('Date').agg({'SE': 'mean'})
    val_seDF = pd.DataFrame({'Minute': val_minutes.values, 'SE': np.square(valY_min_inc)})
    val_seDF['Date'] = val_seDF.Minute.dt.date
    val_mse_by_dateDF = val_seDF.groupby('Date').agg({'SE': 'mean'})
    summaryDF = None
    train_mse = np.sum(np.square(trainY_min_inc))/len(trainY_min_inc)
    val_mse = np.sum(np.square(valY_min_inc))/len(valY_min_inc)
    train_mse_by_dateDF.rename(columns={'SE': 'MSE'}, inplace=True); val_mse_by_dateDF.rename(columns={'SE': 'MSE'}, inplace=True)
    return(train_R2, train_mse, train_mse_by_dateDF, val_mse, val_mse_by_dateDF, summaryDF)


# In[29]:


def simpleLM(trainX, trainY_min_inc, valX, valY_min_inc):
    print('in simpleLM')
    train_minutes, trainX, trainY_min_inc, val_minutes, valX, valY_min_inc =         prepDataForSKModel(trainX, trainY_min_inc, valX, valY_min_inc)
    model = LinearRegression().fit(trainX, trainY_min_inc)
    train_R2, val_R2, train_mse, train_mse_by_dateDF, val_mse, val_mse_by_dateDF, summaryDF =         getSKModelResults(model, trainX, trainY_min_inc, valX, valY_min_inc, train_minutes, val_minutes)
    print('simpleLM gives:   R2='+str(round(train_R2, 5))+'   train_mse='+str(round(train_mse, 5))+'   val_mse='+str(round(val_mse, 5)))
    return(train_R2, train_mse, train_mse_by_dateDF, val_mse, val_mse_by_dateDF, summaryDF)


# In[30]:


def calculateMSEDerivative(lambda_val, trainX, trainY_min_inc, num_k_folds, sk_func, epsilon=.3):
    min_lambda_val = max(0, lambda_val-epsilon)
    max_lambda_val = lambda_val+epsilon
    print('calculating MSE and Derivative for lambda_val='+str(lambda_val))
    mses, mses_minus_epsilon, mses_plus_epsilon = [], [], []
    for k in tqdm.tqdm(range(num_k_folds)):
        OOS_begin = int((k/num_k_folds)*len(trainX))
        OOS_end = int(((k+1)/num_k_folds)*len(trainX))
        OOS_mask = np.zeros(len(trainX))
        OOS_mask[OOS_begin:OOS_end] = 1
        OOS_mask = OOS_mask == 1
        X_OOS = trainX[OOS_mask]
        y_OOS = trainY_min_inc[OOS_mask]
        X_sample = trainX[~OOS_mask]
        y_sample = trainY_min_inc[~OOS_mask]
        model = sk_func(alpha=lambda_val).fit(X_sample, y_sample)
        OOS_preds = model.predict(X_OOS)
        OOS_resid = OOS_preds - trainY_min_inc[OOS_mask]
        mses.append(np.sum(np.square(OOS_resid))/len(OOS_resid))
        model_minus_epsilon = sk_func(alpha=min_lambda_val).fit(X_sample, y_sample)
        OOS_preds = model_minus_epsilon.predict(X_OOS)
        OOS_resid = OOS_preds - trainY_min_inc[OOS_mask]
        mses_minus_epsilon.append(np.sum(np.square(OOS_resid))/len(OOS_resid))
        model_plus_epsilon = sk_func(alpha=max_lambda_val).fit(X_sample, y_sample)
        OOS_preds = model_plus_epsilon.predict(X_OOS)
        OOS_resid = OOS_preds - trainY_min_inc[OOS_mask]
        mses_plus_epsilon.append(np.sum(np.square(OOS_resid))/len(OOS_resid))
    mse = np.mean(mses); mse_minus_epsilon = np.mean(mses_minus_epsilon); mse_plus_epsilon = np.mean(mses_plus_epsilon)
    if (abs(np.sign(mse-mse_minus_epsilon) - np.sign(mse_plus_epsilon-mse)) == 2):
        print('Caution! Both signs point the same way: '+str(np.sign(mse_plus_epsilon-mse)))
    derivative = (mse_plus_epsilon - mse_minus_epsilon)/(max_lambda_val-min_lambda_val)
    return(mse, derivative)

def findLambdaVal(trainX, trainY_min_inc, reg_type='ridge'):
    print('Finding optimal lambda_val')
    assert(reg_type in ['ridge', 'lasso'])
    sk_func = Ridge
    if reg_type=='lasso': sk_funk=Lasso
    lambda_val = 1.0
    learning_rate = .5
    num_k_folds = 5
    mse_tolerance = .0001
    last_mse = 999; curr_mse = 998
    while last_mse - curr_mse > mse_tolerance:
        last_mse = curr_mse
        curr_mse, derivative = calculateMSEDerivative(lambda_val, trainX, trainY_min_inc, num_k_folds, sk_func)
        print('(curr_mse, derivative): '+str((curr_mse, derivative)))
        if derivative==0: return(lambda_val)
        lambda_val -= derivative*learning_rate
    return(lambda_val)



# In[31]:


def normalizeData(trainX, trainY_min_inc, valX, valY_min_inc):
    print('Normalizing data')
    #need to reshape in order to do the scaling. Will unreshape at the end
    trainY_min_inc = trainY_min_inc.reshape(-1, 1); valY_min_inc = valY_min_inc.reshape(-1, 1)
    x_scaler = preprocessing.StandardScaler().fit(trainX)
    y_scaler = preprocessing.StandardScaler().fit(trainY_min_inc)
    trainX_normed = x_scaler.transform(trainX)
    valX_normed = x_scaler.transform(valX)
    trainY_normed = y_scaler.transform(trainY_min_inc)
    valY_normed = y_scaler.transform(valY_min_inc)
    trainY_normed = trainY_normed.reshape(len(trainY_normed)); valY_normed = valY_normed.reshape(len(valY_normed))
    return(trainX_normed, trainY_normed, valX_normed, valY_normed, x_scaler, y_scaler)


# In[32]:


def ridgeLM(trainX, trainY_min_inc, valX, valY_min_inc, lambda_val=None, normalize=False):
    print('in ridgeLM with lambda_val='+str(lambda_val)+' and normalize='+str(normalize))
    train_minutes, trainX, trainY_min_inc, val_minutes, valX, valY_min_inc =         prepDataForSKModel(trainX, trainY_min_inc, valX, valY_min_inc)
    if lambda_val is None: lambda_val = findLambdaVal(trainX, trainY_min_inc, reg_type='ridge')
    model = Ridge(alpha=lambda_val).fit(trainX, trainY_min_inc)
    train_R2, val_R2, train_mse, train_mse_by_dateDF, val_mse, val_mse_by_dateDF, summaryDF =         getSKModelResults(model, trainX, trainY_min_inc, valX, valY_min_inc, train_minutes, val_minutes)
    print('ridgeLM gives:   R2='+str(round(train_R2, 5))+'   train_mse='+str(round(train_mse, 5))+'   val_mse='+str(round(val_mse, 5)))
    return(train_R2, train_mse, train_mse_by_dateDF, val_mse, val_mse_by_dateDF, summaryDF)


# In[33]:


def lassoLM(trainX, trainY_min_inc, valX, valY_min_inc, lambda_val=None, normalize=False):
    print('in lassoLM with lambda_val='+str(lambda_val)+' and normalize='+str(normalize))
    train_minutes, trainX, trainY_min_inc, val_minutes, valX, valY_min_inc =         prepDataForSKModel(trainX, trainY_min_inc, valX, valY_min_inc)
    if lambda_val is None: lambda_val = findLambdaVal(trainX, trainY_min_inc, reg_type='lasso')
    model = Lasso(alpha=lambda_val).fit(trainX, trainY_min_inc)
    train_R2, val_R2, train_mse, train_mse_by_dateDF, val_mse, val_mse_by_dateDF, summaryDF =         getSKModelResults(model, trainX, trainY_min_inc, valX, valY_min_inc, train_minutes, val_minutes)
    print('lassoLM gives:   R2='+str(round(train_R2, 5))+'   train_mse='+str(round(train_mse, 5))+'   val_mse='+str(round(val_mse, 5)))
    return(train_R2, train_mse, train_mse_by_dateDF, val_mse, val_mse_by_dateDF, summaryDF)


# In[34]:


def normRidgeLM(trainX, trainY_min_inc, valX, valY_min_inc, lambda_val=None):
    return(ridgeLM(trainX, trainY_min_inc, valX, valY_min_inc, lambda_val, normalize=True))

def normLassoLM(trainX, trainY_min_inc, valX, valY_min_inc, lambda_val=None):
    return(lassoLM(trainX, trainY_min_inc, valX, valY_min_inc, lambda_val, normalize=True))


# In[35]:


def randomForest(trainX, trainY_min_inc, valX, valY_min_inc):
    print('in randomForest')
    train_minutes, trainX, trainY_min_inc, val_minutes, valX, valY_min_inc =         prepDataForSKModel(trainX, trainY_min_inc, valX, valY_min_inc)
    model = DecisionTreeRegressor(random_state=12).fit(trainX, trainY_min_inc)
    train_R2, val_R2, train_mse, train_mse_by_dateDF, val_mse, val_mse_by_dateDF, summaryDF =         getSKModelResults(model, trainX, trainY_min_inc, valX, valY_min_inc, train_minutes, val_minutes)
    print('randomForest gives:   R2='+str(round(train_R2, 5))+'   train_mse='+str(round(train_mse, 5))+'   val_mse='+str(round(val_mse, 5)))
    return(train_R2, train_mse, train_mse_by_dateDF, val_mse, val_mse_by_dateDF, summaryDF)


# In[36]:


def unnormalizeIndividualData(scale, normed_data):
    means = scale.mean_
    stds = scale.scale_
    unstd_data = np.multiply(normed_data, stds)
    return(np.add(unstd_data, means))

def unnormalizeData(trainX_normed, trainY_normed, valX_normed, valY_normed, x_scaler, y_scaler):
    return(unnormalizeIndividualData(x_scaler, trainX_normed), unnormalizeIndividualData(y_scaler, trainY_normed),
           unnormalizeIndividualData(x_scaler, valY_normed), unnormalizeIndividualData(y_scaler, valY_normed))


# <h4> Keras Models </h4>

# We define these models as a list of dictionaries. The dictionaries can be as follows.
#
# - Dropout
#  - rate
#
# - Dense
#  - activation
#  - units
#

def getHTMLIDFromKerasModelName(keras_model_name):
    out_str = ''
    layer_reps = [i.split('[')[1] for i in keras_model_name.split(']') if i!='']
    for l in layer_reps:
        out_str += l.replace('.', 'dp')+'___'
    return out_str

def getKerasModelNameFromHTMLID(html_id):
    out_str = ''
    layer_reps = [i for i in html_id.split('___') if i!='']
    for l in layer_reps:
        out_str += '['+l.replace('dp', '.')+']'
    return out_str

# In[37]:


def kerasModelArrayToStr(model_array):
    return_str = ''
    for layer_dict in model_array:
        layer_str = 'not set yet'
        if layer_dict['layer'] == 'Dense':
            layer_str = 'Dense_'+str(layer_dict['units'])+'_'+layer_dict['activation'][:3]
        elif layer_dict['layer'] == 'Dropout':
            layer_str = 'Drop_'+str(layer_dict['rate'])
        else:
            raise ValueError(layer_dict['layer']+' not recognized.')
        return_str += '['+layer_str+']'
    return(return_str)

def kerasModelArrayToModel(model_array, input_size):
    assert(model_array[0]['layer'] == 'Dense')
    assert((model_array[-1]['layer'] == 'Dense') and (model_array[-1]['units'] == 1) and (model_array[-1]['activation'] == 'linear') )
    output_model = Sequential()
    output_model.add(Dense(units=model_array[0]['units'], activation=model_array[0]['activation'], input_shape=(input_size,)))
    for layer_dict in model_array[1:]:
        if layer_dict['layer'] == 'Dense':
            output_model.add(Dense(units=layer_dict['units'], activation=layer_dict['activation']))
        elif layer_dict['layer'] == 'Dropout':
            output_model.add(Dropout(rate=layer_dict['rate']))
        else:
            raise ValueError(layer_dict['layer']+' not recognized.')
    output_model.compile(loss='mse', optimizer='adam')
    return(output_model)


# In[38]:


keras_models = [
    [{'layer': 'Dense', 'activation': 'relu', 'units': 64},
        {'layer': 'Dropout', 'rate': .2},
        {'layer': 'Dense', 'activation': 'relu', 'units': 32},
        {'layer': 'Dense', 'activation': 'linear', 'units': 1}],
    [{'layer': 'Dense', 'activation': 'sigmoid', 'units': 64},
        {'layer': 'Dropout', 'rate': .2},
        {'layer': 'Dense', 'activation': 'sigmoid', 'units': 32},
        {'layer': 'Dense', 'activation': 'linear', 'units': 1}],
    [{'layer': 'Dense', 'activation': 'relu', 'units': 256},
        {'layer': 'Dense', 'activation': 'relu', 'units': 128},
        {'layer': 'Dense', 'activation': 'relu', 'units': 64},
        {'layer': 'Dense', 'activation': 'relu', 'units': 32},
        {'layer': 'Dense', 'activation': 'linear', 'units': 1}],
    [{'layer': 'Dense', 'activation': 'relu', 'units': 256},
        {'layer': 'Dense', 'activation': 'relu', 'units': 128},
        {'layer': 'Dropout', 'rate': .1},
        {'layer': 'Dense', 'activation': 'relu', 'units': 64},
        {'layer': 'Dense', 'activation': 'relu', 'units': 32},
        {'layer': 'Dense', 'activation': 'linear', 'units': 1}],
    [{'layer': 'Dense', 'activation': 'relu', 'units': 256},
        {'layer': 'Dense', 'activation': 'relu', 'units': 128},
        {'layer': 'Dropout', 'rate': .2},
        {'layer': 'Dense', 'activation': 'relu', 'units': 64},
        {'layer': 'Dense', 'activation': 'relu', 'units': 32},
        {'layer': 'Dense', 'activation': 'linear', 'units': 1}],
    [{'layer': 'Dense', 'activation': 'relu', 'units': 256},
        {'layer': 'Dense', 'activation': 'relu', 'units': 128},
        {'layer': 'Dropout', 'rate': .2},
        {'layer': 'Dense', 'activation': 'relu', 'units': 64},
        {'layer': 'Dropout', 'rate': .2},
        {'layer': 'Dense', 'activation': 'relu', 'units': 32},
        {'layer': 'Dense', 'activation': 'linear', 'units': 1}]]

keras_model_strs = [kerasModelArrayToStr(keras_model) for keras_model in keras_models]


# In[39]:


def getKerasModelToDoDict(config_dir, trainY, keras_models=keras_models):
    keras_model_strs = [kerasModelArrayToStr(m) for m in keras_models]
    y_min_incs = [int(i.split('y_B')[1]) for i in trainY.columns if i[:3]=='y_B']
    to_do_dict = {};
    for y_min_inc in y_min_incs: to_do_dict[y_min_inc] = keras_model_strs[:]
    done_dict = {}
    print('Searching for current Keras model progress')
    summary_sheet_fn = config_dir+'summary_sheet.html'
    summary_sheet_html = 'not set yet'
    with open(summary_sheet_fn,'r') as fh:
        summary_sheet_html = fh.read()
    # [x] rewrite with soup
    summary_soup = Soup(summary_sheet_html)
    all_models_soup = summary_soup.select('#all_models_table')
    if len(all_models_soup) == 0:
        print('No all_models_soup. Assuming no models have been done.')
    else:
        all_models_table = all_models_soup[0]
        all_modelsDF = pd.read_html(str(all_models_table))[0] #don't know why [0] is necessary
        for y in all_modelsDF.y.unique():
            done_dict[y] = list(all_modelsDF.loc[all_modelsDF.y == y].Name.values)
            done_dict[y] = [m for m in done_dict[y] if m[0]=='['] # only gets Keras models this way
            if len(done_dict[y]) == 0: del done_dict[y]

    for done_y_min_inc_key in done_dict.keys():
        for model_done in done_dict[done_y_min_inc_key]:
            if model_done in to_do_dict[done_y_min_inc_key]:
                to_do_dict[done_y_min_inc_key].remove(model_done)
            else:
                print('\nWeird! For y_minute='+str(y_min_inc)+' we have done an extra model: '+model_done+'\n')
    if len(done_dict.keys()) == 0:
        print('\nNo models found. Models to go: '+str(to_do_dict)+'...')
    else:
        print('Some models already done: '+str(done_dict)+'\n\nModels to go: '+str(to_do_dict))
    return(to_do_dict)

# getKerasModelToDoDict(config_dir, trainY)


# In[40]:
def getMSEInfoByChunking(train_preds, trainY_min_inc, val_preds, valY_min_inc, train_minutes, val_minutes):
    """
    Obtain the MSE Info using chunking because we are running into MemoryErrors otherwise.
    """
    chunk_size = 2048*2*2
    train_seDF = pd.DataFrame(columns=['Date', 'SE_sum', 'count'])
    val_seDF = pd.DataFrame(columns=['Date', 'SE_sum', 'count'])
    assert train_preds.shape==trainY_min_inc.shape, 'train_preds.shape='+str(train_preds.shape)+' trainY_min_inc.shape='+str(trainY_min_inc.shape)
    assert val_preds.shape==valY_min_inc.shape, 'val_preds.shape='+str(val_preds.shape)+' valY_min_inc.shape='+str(valY_min_inc.shape)
    print('train_preds.shape: '+str(train_preds.shape))
    num_chunks = int(np.ceil(len(train_preds)/chunk_size))
    print('chunking train...')
    for i in range(num_chunks):
        cs = i*chunk_size; ce = (i+1)*chunk_size #chunk start and chunk end
        print(train_minutes.values[cs:ce][0])
        chunk_trainDF = pd.DataFrame({'Minute': train_minutes.values[cs:ce],
                                      'Pred': train_preds[cs:ce],
                                      'TrueY': trainY_min_inc[cs:ce]})
        chunk_trainDF['Resid'] = chunk_trainDF.Pred - chunk_trainDF.TrueY
        chunk_trainDF['SE'] = chunk_trainDF.Resid * chunk_trainDF.Resid
        chunk_trainDF['Date'] = chunk_trainDF.Minute.dt.date
        chunk_agg = chunk_trainDF.groupby('Date').agg({'SE': 'sum', 'Pred': 'count'})
        chunk_agg['Date'] = chunk_agg.index
        chunk_agg.rename(columns={'SE': 'SE_sum', 'Pred': 'count'}, inplace=True)
        train_seDF = train_seDF.append(chunk_agg, ignore_index=True)
    num_chunks = int(np.ceil(len(val_preds)/chunk_size))
    print('chunking val...')
    for i in range(num_chunks):
        cs = i*chunk_size; ce = (i+1)*chunk_size #chunk start and chunk end
        print(val_minutes.values[cs:ce][0])
        chunk_valDF = pd.DataFrame({'Minute': val_minutes.values[cs:ce],
                                      'Pred': val_preds[cs:ce],
                                      'TrueY': valY_min_inc[cs:ce]})
        chunk_valDF['Resid'] = chunk_valDF.Pred - chunk_valDF.TrueY
        chunk_valDF['SE'] = chunk_valDF.Resid * chunk_valDF.Resid
        chunk_valDF['Date'] = chunk_valDF.Minute.dt.date
        chunk_agg = chunk_valDF.groupby('Date').agg({'SE': 'sum', 'Pred': 'count'})
        chunk_agg['Date'] = chunk_agg.index
        chunk_agg.rename(columns={'SE': 'SE_sum', 'Pred': 'count'}, inplace=True)
        val_seDF = val_seDF.append(chunk_agg, ignore_index=True)
    train_mse = train_seDF.SE_sum.sum()/train_seDF['count'].sum()
    val_mse = val_seDF.SE_sum.sum()/val_seDF['count'].sum()
    train_mse_by_dateDF = train_seDF.groupby('Date').agg({'SE_sum': 'sum', 'count': 'sum'})
    train_mse_by_dateDF['Date'] = train_mse_by_dateDF.index
    train_mse_by_dateDF['MSE'] = train_mse_by_dateDF.SE_sum/train_mse_by_dateDF['count']
    val_mse_by_dateDF = val_seDF.groupby('Date').agg({'SE_sum': 'sum', 'count': 'sum'})
    val_mse_by_dateDF['Date'] = val_mse_by_dateDF.index
    val_mse_by_dateDF['MSE'] = val_mse_by_dateDF.SE_sum/val_mse_by_dateDF['count']
    return(train_mse, train_mse_by_dateDF[['Date', 'MSE']], val_mse, val_mse_by_dateDF[['Date', 'MSE']])

def getKerasModelResults(keras_model, trainX, trainY_min_inc, valX, valY_min_inc, train_minutes, val_minutes):
    print('in getKerasModelResults')
    summaryDF = None
    train_preds = keras_model.predict(trainX)
    train_preds = train_preds.reshape(len(train_preds))
    val_preds = keras_model.predict(valX)
    val_preds = val_preds.reshape(len(val_preds))
    train_R2 = r2_score(trainY_min_inc, train_preds)
    val_R2 = r2_score(valY_min_inc, val_preds)
#     train_resid = train_preds - trainY_min_inc
#     val_resid = val_preds - valY_min_inc
#     train_mse = np.sum(np.square(train_resid))/len(train_preds)
#     val_mse = np.sum(np.square(val_resid))/len(val_preds)
#     train_seDF = pd.DataFrame({'Minute': train_minutes.values, 'SE': np.square(train_resid)})
#     train_seDF['Date'] = train_seDF.Minute.dt.date
#     train_mse_by_dateDF = train_seDF.groupby('Date').agg({'SE': 'mean'})
#     val_seDF = pd.DataFrame({'Minute': val_minutes.values, 'SE': np.square(val_resid)})
#     val_seDF['Date'] = val_seDF.Minute.dt.date
#     val_mse_by_dateDF = val_seDF.groupby('Date').agg({'SE': 'mean'})
#     train_mse_by_dateDF.rename(columns={'SE': 'MSE'}, inplace=True); val_mse_by_dateDF.rename(columns={'SE': 'MSE'}, inplace=True)
    train_mse, train_mse_by_dateDF, val_mse, val_mse_by_dateDF = \
        getMSEInfoByChunking(train_preds, trainY_min_inc, val_preds, valY_min_inc, train_minutes, val_minutes)
    return(train_R2, val_R2, train_mse, train_mse_by_dateDF, val_mse, val_mse_by_dateDF, summaryDF)


# In[41]:


def isMonotonicIncreasing(input_array):
    return(all(input_array[i] <= input_array[i + 1] for i in range(len(input_array) - 1)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[42]:


def trainKerasModel(keras_model, model_str, trainX, trainY_min_inc, y_min_inc, train_minutes, config_dir):
    """
    1. Splits train data into a validation set
    2. Trains the model, using an algorithm to decide when to stop training
    3. Saves progress along the way
    """
    print('in trainKerasModel')
    y_min_inc_dir = config_dir + 'Deep_Models/'+str(y_min_inc)+'/'
    model_dir = y_min_inc_dir+model_str+'/'
    starting_epoch, epoch = 0, 0
    if not os.path.exists(y_min_inc_dir): os.mkdir(y_min_inc_dir)
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    if len(os.listdir(model_dir)) > 0:
        starting_epoch = max(int(fn.split('.')[0]) for fn in os.listdir(model_dir))+1
        epoch = starting_epoch + 0
    # [x] 1. Splits train data into a validation set
    split_pct = .7
    dates = train_minutes.dt.date.unique()
    tr_dates = dates[:int(split_pct*len(dates))]
    tr_mask = train_minutes.dt.date.isin(tr_dates)
    tr_X = trainX[tr_mask]; tr_Y = trainY_min_inc[tr_mask]
    vl_X = trainX[~tr_mask]; vl_Y = trainY_min_inc[~tr_mask]
    # [x] 2. Trains the model, using an algorithm to decide when to stop training
    tr_mses, vl_mses, = [],[]

    min_epochs = 5
    max_epochs = 50
    num_prev_models_to_consider = 3

    if starting_epoch > 0:
        print('reconstructing '+str(starting_epoch)+' vl_mses')
        # [x] reconstruct vl_mses
        epochs_to_reconstruct = range(starting_epoch)
        for ep in epochs_to_reconstruct:
            reconstructed_model_fn = model_dir+str(ep)+'.h5'
            reconstructed_model = load_model(reconstructed_model_fn)
            reconstructed_model_vl_mse = reconstructed_model.evaluate(vl_X, vl_Y)
            vl_mses.append((ep, reconstructed_model_vl_mse))
        print(vl_mses)

    # If we notice val_mse is increasing across num_prev_models_to_consider, we stop and return.
    # Else, return the min val_mse model
    while epoch < max_epochs:
        if epoch > 0: keras_model = load_model(model_dir+str(epoch-1)+'.h5') # I think we need to do this but not sure
        if epoch % 3 == 0: print('\nepoch: '+str(epoch))
        history = keras_model.fit(tr_X, tr_Y, validation_data=(vl_X, vl_Y), shuffle=False, epochs=1, verbose=1)
        vl_mse = history.history['val_loss'][0]
        vl_mses.append((epoch, vl_mse))
#         tr_mse = keras_model.evaluate(tr_X, tr_Y)
        model_fn = model_dir+str(epoch)+'.h5'
        keras_model.save(model_fn)
        if len(vl_mses) > 2*num_prev_models_to_consider:
            if isMonotonicIncreasing([i[1] for i in vl_mses[-num_prev_models_to_consider:]]):
                print('val mse has been monotonic increasing for the past '+str(num_prev_models_to_consider)+' epochs. Breaking.')
                break
            min_idx = np.argmin([t[1] for t in vl_mses])
            if epoch - min_idx > num_prev_models_to_consider:
                print('We saw the best vl_mse '+str(epoch - min_idx)+' epochs ago. Breaking.')
                break
        epoch += 1
    # return min model in stack
    min_idx = np.argmin([t[1] for t in vl_mses])
    print('The best model is from epoch '+str(vl_mses[min_idx][0])+' with a val_mse of '+str(vl_mses[min_idx][1]))
    best_model = load_model(model_dir+str(vl_mses[min_idx][0])+'.h5')
    return(best_model)


# In[43]:


def kerasModel(model_array, trainX, trainY_min_inc, y_min_inc, valX, valY_min_inc, config_dir):
    model_str = kerasModelArrayToStr(model_array)
    print('in kerasModel with: '+model_str)
    train_minutes, trainX, trainY_min_inc, val_minutes, valX, valY_min_inc =         prepDataForSKModel(trainX, trainY_min_inc, valX, valY_min_inc)
    keras_model = kerasModelArrayToModel(model_array, input_size=trainX.shape[1])
    keras_model = trainKerasModel(keras_model, model_str, trainX, trainY_min_inc, y_min_inc, train_minutes, config_dir)
    train_R2, val_R2, train_mse, train_mse_by_dateDF, val_mse, val_mse_by_dateDF, summaryDF =         getKerasModelResults(keras_model, trainX, trainY_min_inc, valX, valY_min_inc, train_minutes, val_minutes)
    print('kerasModel gives:   R2='+str(round(train_R2, 5))+'   train_mse='+str(round(train_mse, 5))+'   val_mse='+str(round(val_mse, 5)))
    return(train_R2, train_mse, train_mse_by_dateDF, val_mse, val_mse_by_dateDF, summaryDF)


# In[44]:


def graphDailyMSE(mse_by_dateDF, zero_model_mse_by_dateDF, graph_title='untitled'):
    assert np.all(zero_model_mse_by_dateDF.index.values == mse_by_dateDF.index.values), 'Dates do not match up with zeromodel for graphing.'
    mergeDF = pd.merge(mse_by_dateDF, zero_model_mse_by_dateDF, suffixes=('_model', '_zero'), left_index=True, right_index=True)
    return(mergeDF.plot(title=graph_title))


# In[45]:


def findMod(bigger_soup, elem_type, search_string):
    """
    Guards against not finding the element in find() because of newlines
    """
    elems = bigger_soup.find_all(elem_type)
    for elem in elems:
        if search_string in elem.string:
            return elem
    return None


# In[46]:


def updateResultsSummary(summary_soup, model_name, config_dir, y_min_inc,
                       train_R2, train_mse, train_mse_by_dateDF, val_mse, val_mse_by_dateDF, summaryDF,
                      zm_train_R2, zm_train_mse, zm_train_mse_by_dateDF, zm_val_mse, zm_val_mse_by_dateDF):
    print('in updateResultsSummary')
    res_sum = findMod(summary_soup, 'h1', 'Results Summary')
    config_soup = findMod(summary_soup, 'h1', 'Configuration')
    best_models = summary_soup.select("#best_models")
    if len(best_models) == 0:
        best_models = summary_soup.new_tag('h3', id='best_models')
        best_models.string = 'Best Models'
        res_sum.insert_after(best_models)
    else:
        best_models = best_models[0]

    best_models_table  = summary_soup.select("#best_models_table")
    new_potential_row =  {'y':[y_min_inc], 'Name':[model_name], '% Imprv ZM':[round((zm_val_mse-val_mse)/zm_val_mse, 3)], 'Val MSE':[round(val_mse, 5)], 'ZM Val MSE':[round(zm_val_mse, 5)], 'Train R2':[round(train_R2, 5)], 'Val R2':[np.nan], 'Train MSE':[round(train_mse,4)], 'ZM Train MSE':[round(zm_train_mse,4)]}
    col_order = pd.DataFrame(new_potential_row).columns.values
    if len(best_models_table) == 0:
        # [x] create new best_models_table
        print('No best models table found. Creating best models')
        best_modelsDF = pd.DataFrame(new_potential_row)
        best_models_html = best_modelsDF.to_html(table_id = 'best_models_table', index=False, col_space=50, justify='center')
        best_models_soup = Soup(best_models_html)
        best_models.insert_after(best_models_soup)
        best_models_soup = summary_soup.select('#best_models_table')[0] #I don't know why we have to redefine this after the insert but we do
        # [x] create plots div
        best_plots_div = summary_soup.new_tag('div', id='best_plots')
        best_models_soup.insert_after(best_plots_div)
        model_plot = summary_soup.new_tag('img', id='min'+str(y_min_inc)+'_plot',
                                  src=config_dir+'Plots/'+model_name+'_'+str(y_min_inc)+'min_val_MSE.png',
                                  height='400')
        best_plots_div.append(model_plot)
    else:
        best_models_table = best_models_table[0]
        best_modelsDF = pd.read_html(str(best_models_table))[0] #don't know why the [0] is necessary
        # [x] update existing best_models_table
        y_min_inc_subDF = best_modelsDF.loc[best_modelsDF.y == y_min_inc]
        best_plots_div = summary_soup.select('#best_plots')[0]
        if len(y_min_inc_subDF) == 0: #we have no best_model for this y_min_inc yet
            best_modelsDF = best_modelsDF.append(pd.DataFrame(new_potential_row), ignore_index=True)
            # [x] add plot
            model_plot = summary_soup.new_tag('img', id='min'+str(y_min_inc)+'_plot',
                                      src=config_dir+'Plots/'+model_name+'_'+str(y_min_inc)+'min_val_MSE.png',
                                      height='400')
            best_plots_div.append(model_plot)
            # re-sort plots accd to y_min_inc?
        else:
            best_val_mse_imprv = y_min_inc_subDF['% Imprv ZM'].iloc[0]
            if new_potential_row['% Imprv ZM'] > best_val_mse_imprv:
                print('Updating best models for y='+str(y_min_inc)+': '+
                      new_potential_row['Name'][0]+' ('+str(new_potential_row['Val MSE'][0])+') beats '+
                      y_min_inc_subDF.Name.iloc[0]+' ('+str(y_min_inc_subDF['Val MSE'].values[0])+')')
                # [x] update table
                best_modelsDF = best_modelsDF.loc[best_modelsDF.y != y_min_inc]
                best_modelsDF = best_modelsDF.append(pd.DataFrame(new_potential_row), ignore_index=True)
                # [x] update plots
                y_min_inc_plot = summary_soup.select('#min'+str(y_min_inc)+'_plot')[0]
                y_min_inc_plot['src']=config_dir+'Plots/'+model_name+'_'+str(y_min_inc)+'min_val_MSE.png' #this update step isn't working
        # re-sort DF and replace the html
        print(best_modelsDF)
        best_modelsDF = best_modelsDF.sort_values('y').reset_index(drop=True)[col_order]
        best_models_html = best_modelsDF.to_html(table_id = 'best_models_table', index=False, col_space=50, justify='center')
        best_models_soup = Soup(best_models_html, 'html.parser')
        best_models_table.replace_with(best_models_soup)
    # All Models
    all_models_table = summary_soup.select("#all_models_table")
    if len(all_models_table) == 0:
        # [x] create All Models if it doesn't exist
        print('No all_models_table found. Creating.')
        all_models = summary_soup.new_tag('h3', id='all_models')
        all_models.string = 'All Models'
        config_soup = findMod(summary_soup, 'h1', 'Configuration')
        config_soup.insert_before(all_models)
        all_models = summary_soup.select('#all_models')[0] #I don't know why we have to redefine this after the insert but we do
        all_modelsDF = pd.DataFrame(new_potential_row)
        all_models_html = all_modelsDF.to_html(table_id = 'all_models_table', index=False, col_space=50, justify='center')
        all_models_soup = Soup(all_models_html)
        all_models.insert_after(all_models_soup)
    else:
        # [x] add to All Models
        all_models_table = all_models_table[0]
        all_modelsDF = pd.read_html(str(all_models_table))[0] #don't know why the [0] is necessary
        all_modelsDF = all_modelsDF.append(pd.DataFrame(new_potential_row), ignore_index=True)
        all_modelsDF.iloc[all_modelsDF['Name'].map(model_order_dict).argsort()] #sort first by model_name
        all_modelsDF = all_modelsDF.sort_values('y').reset_index(drop=True)[col_order]
        all_models_soup = Soup(all_modelsDF.to_html(table_id = 'all_models_table', index=False, col_space=50, justify='center'))
        all_models_table.replace_with(all_models_soup)
    # print(summary_soup.prettify())
    return(summary_soup)


# In[47]:


def updateModelDetails(summary_soup, model_name, config_dir, y_min_inc,
                       train_R2, train_mse, train_mse_by_dateDF, val_mse, val_mse_by_dateDF, summaryDF,
                      zm_train_R2, zm_train_mse, zm_train_mse_by_dateDF, zm_val_mse, zm_val_mse_by_dateDF):
    print('in updateModelDetails')
    # [x] create model results html
    html_id_friendly_model_name = model_name if model_name[0] != '[' else getHTMLIDFromKerasModelName(model_name)

    name_tag = summary_soup.new_tag('h4', id=html_id_friendly_model_name+str(y_min_inc))
    name_tag.string=model_name
    new_row =  {'y':[y_min_inc], 'Name':[model_name], '% Imprv ZM':[round((zm_val_mse-val_mse)/zm_val_mse, 3)], 'Val MSE':[round(val_mse, 5)], 'ZM Val MSE':[round(zm_val_mse, 5)], 'Train R2':[round(train_R2, 5)], 'Val R2':[np.nan], 'Train MSE':[round(train_mse,4)], 'ZM Train MSE':[round(zm_train_mse,4)]}
    results_table_html = pd.DataFrame(new_row).to_html(index=False, col_space=50, justify='center')
    # [x] plot
    model_plot = summary_soup.new_tag('img',
                                  src=config_dir+'Plots/'+model_name+'_'+str(y_min_inc)+'min_val_MSE.png',
                                  height='400')
    # [x] summary
    has_summary = (summaryDF is not None)
    model_summary_html = summaryDF.to_html() if has_summary else '<p> No model summaryDF available </p>'
    model_summary_soup = Soup(model_summary_html)
    if has_summary:
        # make the sumamry table scrollable
        model_summary_soup.thead['style'] = 'display:block;'
        model_summary_soup.tbody['style'] = 'height:300px; overflow-y:scroll; display:block;'
    min_chg_h3 = findMod(summary_soup, 'h3', str(y_min_inc)+' Minute Chg')
    if min_chg_h3 is None:
        # [x] create h3
        new_min_chg_h3_tag = summary_soup.new_tag('h3')
        new_min_chg_h3_tag.string=str(y_min_inc)+' Minute Chg'
        #insert new h3 to the end
        summary_soup.body.insert(len(summary_soup.body.contents), new_min_chg_h3_tag)
        new_min_chg_h3_tag = findMod(summary_soup, 'h3', str(y_min_inc)+' Minute Chg')
        new_min_chg_h3_tag.insert_after(name_tag)
    else:
        # [x] add model results onto the end of h3
        next_h3 = min_chg_h3.find_next_sibling('h3')
        if next_h3 is None:
            #just add to the end of the document. We are at the end.
            summary_soup.body.insert(len(summary_soup.body.contents), name_tag)
        else:
            next_h3.insert_before(name_tag)
    # find name_tag and insert results table, plot, and summary
    name_tag = summary_soup.select('#'+html_id_friendly_model_name+str(y_min_inc))[0]
    name_tag.insert_after(model_summary_soup)
    name_tag = summary_soup.select('#'+html_id_friendly_model_name+str(y_min_inc))[0]
    name_tag.insert_after(model_plot)
    name_tag = summary_soup.select('#'+html_id_friendly_model_name+str(y_min_inc))[0]
    name_tag.insert_after(Soup(results_table_html))
    return summary_soup



# In[48]:


def recordModelResults(model_name, config_dir, y_min_inc,
                       train_R2, train_mse, train_mse_by_dateDF, val_mse, val_mse_by_dateDF, summaryDF,
                      zm_train_R2, zm_train_mse, zm_train_mse_by_dateDF, zm_val_mse, zm_val_mse_by_dateDF):
    print('recording '+model_name+' results')
    model_plot = graphDailyMSE(val_mse_by_dateDF, zm_val_mse_by_dateDF, graph_title='y='+str(y_min_inc)+' Val: '+model_name+' vs ZM')
    model_plot.figure.savefig(config_dir+'Plots/'+model_name+'_'+str(y_min_inc)+'min_val_MSE.png', dpi=256)
    summary_sheet_fn = config_dir+'summary_sheet.html'
    summary_sheet_html = 'not set yet'
    with open(summary_sheet_fn,'r') as fh:
        summary_sheet_html = fh.read()
    summary_soup = Soup(summary_sheet_html)
    # [x] update Results Summary
    summary_soup = updateResultsSummary(summary_soup, model_name, config_dir, y_min_inc,
                       train_R2, train_mse, train_mse_by_dateDF, val_mse, val_mse_by_dateDF, summaryDF,
                      zm_train_R2, zm_train_mse, zm_train_mse_by_dateDF, zm_val_mse, zm_val_mse_by_dateDF)
    # [] update Model Details
    summary_soup = updateModelDetails(summary_soup, model_name, config_dir, y_min_inc,
                       train_R2, train_mse, train_mse_by_dateDF, val_mse, val_mse_by_dateDF, summaryDF,
                      zm_train_R2, zm_train_mse, zm_train_mse_by_dateDF, zm_val_mse, zm_val_mse_by_dateDF)
    print('Rewriting html to '+summary_sheet_fn)
    with open(summary_sheet_fn,'w') as fh:
        fh.write(summary_soup.prettify())



# In[68]:


# progressKerasModels(trainX, trainY, valX, valY, config_dir, keras_models)


# In[176]:


def progressModels(trainX, trainY, valX, valY, config_dir):
#     'simple_LM', 'ridge_LM', 'norm_ridge_LM', 'lasso_LM', norm_lasso_LM', 'random_forest', 'XGBoost'
    func_dict = {'ZM': zeroModel, 'simple_LM': simpleLM, 'ridge_LM': ridgeLM, 'lasso_LM': lassoLM,
                 'norm_ridge_LM': normRidgeLM, 'norm_lasso_LM': normLassoLM}#, 'random_forest': randomForest}
    to_do_dict = getModelToDoDict(config_dir, trainY)
    y_min_incs_to_go = sorted(to_do_dict.keys())
    for y_min_inc in y_min_incs_to_go:
        print('\n'+'='*7+' y Minute '+str(y_min_inc)+' '+'='*7)
        trainY_min_inc, valY_min_inc = formatYData(trainY, valY, y_min_inc)
        zm_train_R2, zm_train_mse, zm_train_mse_by_dateDF, zm_val_mse, zm_val_mse_by_dateDF, _ =             zeroModel(trainX, trainY_min_inc, valX, valY_min_inc)
        for model_to_go in to_do_dict[y_min_inc]:
            if model_to_go in func_dict.keys():
                train_R2, train_mse, train_mse_by_dateDF, val_mse, val_mse_by_dateDF, summaryDF =                     func_dict[model_to_go](trainX, trainY_min_inc, valX, valY_min_inc) # finds the appropriate fxn and calls it
                recordModelResults(model_to_go, config_dir, y_min_inc,
                                   train_R2, train_mse, train_mse_by_dateDF, val_mse, val_mse_by_dateDF, summaryDF,
                                  zm_train_R2, zm_train_mse, zm_train_mse_by_dateDF, zm_val_mse, zm_val_mse_by_dateDF)
            else:
                print('ALERT!: '+model_to_go+' not defined in func_dict!')



# In[50]:


def progressKerasModels(trainX, trainY, valX, valY, config_dir, keras_models):
    keras_model_strs = [kerasModelArrayToStr(m) for m in keras_models]
    to_do_dict = getKerasModelToDoDict(config_dir, trainY)
    y_min_incs_to_go = sorted(to_do_dict.keys())
    for y_min_inc in y_min_incs_to_go:
        print('\n'+'='*7+' y Minute '+str(y_min_inc)+' '+'='*7)
        trainY_min_inc, valY_min_inc = formatYData(trainY, valY, y_min_inc)
        zm_train_R2, zm_train_mse, zm_train_mse_by_dateDF, zm_val_mse, zm_val_mse_by_dateDF, _ =             zeroModel(trainX, trainY_min_inc, valX, valY_min_inc)
        for model_to_go in to_do_dict[y_min_inc]:
            model_array = keras_models[keras_model_strs.index(model_to_go)]
            train_R2, train_mse, train_mse_by_dateDF, val_mse, val_mse_by_dateDF, summaryDF =                 kerasModel(model_array, trainX, trainY_min_inc, y_min_inc, valX, valY_min_inc, config_dir)
            recordModelResults(model_to_go, config_dir, y_min_inc,
                                   train_R2, train_mse, train_mse_by_dateDF, val_mse, val_mse_by_dateDF, summaryDF,
                                  zm_train_R2, zm_train_mse, zm_train_mse_by_dateDF, zm_val_mse, zm_val_mse_by_dateDF)




# In[45]:


def iterateLearning(sec, combDF, omni_dir=omni_dir, email_progress=False):
    """
    1. Creates the file structure if it doesn't exist.
    2. Locates or creates the data in the appropriate format
    3. Finds the most recent model progress, if any, and progresses remaining models.
    4. Saves results
    5. Optionally emails results
    """
    comb_subDF = combDF.loc[combDF.Sec1 == sec].reset_index(drop=True)
    print('='*50+'\nIterating learning on '+sec+' with '+str(len(comb_subDF))+' rows:\n\n'+str(comb_subDF.head())+'\n'+'='*50)
    for i in comb_subDF.index:
        # [x] 1. Creates the file structure if it doesn't exist.
        config_dir, data_dir, summary_sheet_fn, progress_notes_fn, log_fn, deep_models_dir = locateConfigDir(comb_subDF.loc[i], omni_dir)
        # [x] 2. Locates or creates the data in the appropriate format
        trainX, trainY, valX, valY = loadData(comb_subDF.loc[i], omni_dir, config_dir, tdm_dir+'Minute_Files/')
        # [x] 3. Finds the most recent model progress, if any, and progresses remaining models.
        # [x] 4. Saves results
        progressModels(trainX, trainY, valX, valY, config_dir)


# In[67]:


idx = 190
config_dir, data_dir, summary_sheet_fn, progress_notes_fn, log_fn, deep_models_dir = locateConfigDir(combDF.iloc[idx], omni_dir)
trainX, trainY, valX, valY = loadData(combDF.iloc[idx], omni_dir, config_dir, tdm_dir+'Minute_Files/')


# In[53]:


trainY_min_inc, valY_min_inc = formatYData(trainY, valY, 1)


# In[ ]:

progressModels(trainX, trainY, valX, valY, config_dir)
progressKerasModels(trainX, trainY, valX, valY, config_dir, keras_models)



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:
