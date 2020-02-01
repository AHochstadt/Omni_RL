from utils import in_notebook, utcToChi, getChiTimeNow

import pandas as pd
import numpy as np

import os, pytz, datetime

from bs4 import BeautifulSoup as Soup

if in_notebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm



omni_dir = '/home/andrew/All_Trading/Studies/Omni_Project/'
tdm_dir = '/media/andrew/FreeAgent Drive/Market_Data/Tick_Data_Manager/'

# Create directory structure if it doesn't exist

def createConfigDir(comb_row, sec1_dir, day_chg_incs, minute_incs):
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
    postprocessed_data_dir = config_dir+'Postprocessed_Data/'
    os.mkdir(postprocessed_data_dir)

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

        progress_notes_file.write('day_chg_incs: '+str(day_chg_incs)+'\n')
        progress_notes_file.write('minute_incs: '+str(minute_incs)+'\n')

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
                                 '<br>ValEndDate: '+comb_row.ValEndDate.strftime(format='%Y-%m-%d')+

                                 '<br>day_chg_incs: '+str(day_chg_incs)+
                                 '<br>minute_incs: '+str(minute_incs)+'</p>')
        # [x] save config table
        # [x] specify line_width in to_html below
        summary_sheet_file.write(summaryDF.to_html(col_space=200))
        summary_sheet_file.write('<h1>Model Details</h1>')
    print('Config file structure creation complete.\n')
    return config_dir

def locateConfigDir(comb_row, omni_dir, day_chg_incs, minute_incs):
    sec1_dir = omni_dir + 'Primary_Assets/' + comb_row.Sec1 +'/'
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
            # [x] Identify whether this is the correct configuration
            day_chg_incs_cfg = config_section.split('day_chg_incs: ')[1].split('\n')[0]
            minute_incs_cfg = config_section.split('minute_incs: ')[1].split('\n')[0]

            train_start_date = config_section.split('TrainStartDate: ')[1].split('\n')[0]
            val_start_date = config_section.split('ValStartDate: ')[1].split('\n')[0]
            val_end_date = config_section.split('ValEndDate: ')[1].split('\n')[0]
            other_secs = config_section.split('other_secs: ')[1].split('\n')[0].split(',')
            comb_row_other_secs_cols = [col for col in comb_row.index if (col[:3]=='Sec' and col != 'Sec1')]
            comb_row_other_secs = comb_row[comb_row_other_secs_cols]
            comb_row_other_secs = [sec for sec in comb_row_other_secs if type(sec)==str]
            if (str(day_chg_incs) == day_chg_incs_cfg and
                str(minute_incs) == minute_incs_cfg and
                comb_row.TrainStartDate.strftime(format='%Y-%m-%d') == train_start_date and
                comb_row.ValStartDate.strftime(format='%Y-%m-%d') == val_start_date and
                comb_row.ValEndDate.strftime(format='%Y-%m-%d') == val_end_date and
                set(comb_row_other_secs) == set(other_secs)):
                config_dir = sec1_dir+config+'/'
                print('Existing matching config dir found: '+sec1_dir+config)
                return config_dir
    return createConfigDir(comb_row, sec1_dir, day_chg_incs, minute_incs)

### Format/Process Data

def getExpectedPostprocessedCols(num_secs, day_chg_incs, minute_incs):
    # there is probably a more elegant way to do this
    return_cols = ['Minute']
    # outer loop is always SEC for [DAY], [MINUTE] is always the outer loop for [SEC]. ...I think
    # [SEC1] means starts at sec1, [SEC2] means starts at sec2
    chunk1 = ['O_B[SEC1]', 'O_A[SEC1]', 'H_B[SEC1]', 'H_A[SEC1]', 'L_B[SEC1]', 'L_A[SEC1]', 'C_B[SEC1]', 'C_A[SEC1]',
              'Count[SEC1]', 'B_TickImb[SEC1]', 'A_TickImb[SEC1]', 'M_TickImb[SEC1]']
    chunk2 = ['Sec[SEC1]_Open_B', 'Sec[SEC1]_Open_A',
              'Sec[SEC1]_Open_B_chg[DAY]', 'Sec[SEC1]_Open_A_chg[DAY]']
    chunk3 = ['Sec[SEC2]_Open_B_Quotient', 'Sec[SEC2]_Open_A_Quotient']
    chunk4 = ['O_B[SEC1]_ema[MINUTE]', 'O_A[SEC1]_ema[MINUTE]', 'H_B[SEC1]_ema[MINUTE]', 'H_A[SEC1]_ema[MINUTE]', 'L_B[SEC1]_ema[MINUTE]', 'L_A[SEC1]_ema[MINUTE]', 'C_B[SEC1]_ema[MINUTE]', 'C_A[SEC1]_ema[MINUTE]']
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

def getTrainValDates(config_dir):
    summary_sheet_fn = config_dir+'summary_sheet.html'
    summary_sheet_html = ''
    with open(summary_sheet_fn, 'r') as fh:
        summary_sheet_html = fh.read()

    train_start_date = summary_sheet_html.split('TrainStartDate: ')[1][:10]
    val_start_date = summary_sheet_html.split('ValStartDate: ')[1][:10]
    val_end_date = summary_sheet_html.split('ValEndDate: ')[1][:10]

    return train_start_date, val_start_date, val_end_date


def postprocessData(all_minutesDF, dailyDF, sec_guideDF, config_dir, day_chg_incs, minute_incs, saveProcessed=True):
    print('in postprocessData')
    all_minutesDF_orig = all_minutesDF.copy()
    _, val_start_date, _ = getTrainValDates(config_dir) #might have to convert into an actual date object.
    val_start_date = pd.to_datetime(val_start_date).date()
    num_secs = len(sec_guideDF)

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

    # [x] - change on day_chg_incs days
    dailyDF = dailyDF.loc[pd.to_datetime(dailyDF.Date).isin(all_minutesDF.Date)].reset_index(drop=True) #I'm 97% sure this line is ok. Avoids nans in the secX_Open etc columns

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

    # if the first day doesn't have an Open_A or Close_A for one of the securities, drop it
    first_date = dailyDF.Date.iloc[0]
    for sec_num in range(1, num_secs+1):
        if (all(dailyDF['Sec'+str(sec_num)+'_Open_B'].loc[dailyDF.Date == first_date].isna()) or
            all(dailyDF['Sec'+str(sec_num)+'_Open_A'].loc[dailyDF.Date == first_date].isna())):
            print('dropping first_date', first_date)
            dailyDF = dailyDF.loc[dailyDF.Date != first_date].reset_index(drop=True)
            all_minutesDF = all_minutesDF.loc[all_minutesDF.Minute.dt.date != first_date].reset_index(drop=True)

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

    # [x] - last minute_incs minutes
    print('getting data for last '+str(minute_incs)+' minutes')
    col_stems_to_add = ['Count', 'B_TickImb', 'A_TickImb', 'M_TickImb']
    col_stems_to_ema = ['O_B', 'O_A', 'H_B', 'H_A', 'L_B', 'L_A', 'C_B', 'C_A']
    cols_to_add, cols_to_ema = [], []
    for sec_num in range(1, num_secs+1):
        cols_to_add += [s+str(sec_num) for s in col_stems_to_add]
        cols_to_ema += [s+str(sec_num) for s in col_stems_to_ema]
    new_cols_add, new_cols_ema = [], []
    for min_inc in minute_incs:
        new_cols_add += [col+'_sum'+str(min_inc) for col in cols_to_add]
        new_cols_ema += [col+'_ema'+str(min_inc) for col in cols_to_ema]

    # ema the ema columns, then fill in the first min_inc rows of each date with the BOD value
    print('creating minute ema cols...')
    for min_inc in minute_incs:
        all_minutesDF[[col+'_ema'+str(min_inc) for col in cols_to_ema]] = all_minutesDF[cols_to_ema].ewm(com=min_inc).mean()
        for date in tqdm(all_minutesDF.Date.unique()):
            date_subDF = all_minutesDF.loc[all_minutesDF.Date == date]
            repl_subDF = date_subDF.iloc[:min_inc]
            all_minutesDF.loc[repl_subDF.index, [col+'_ema'+str(min_inc) for col in cols_to_ema]] = \
                repl_subDF[cols_to_ema].iloc[0].values

    # add the add columns, making sure not to go past the current date
    print('creating minute add cols...')
    for min_inc in minute_incs:
        for col in cols_to_add: all_minutesDF[col+'_sum'+str(min_inc)] = np.nan
    for date in tqdm(all_minutesDF.Date.unique()):
        date_subDF = all_minutesDF.loc[all_minutesDF.Date == date]
        for min_inc in minute_incs:
            all_minutesDF.loc[date_subDF.index, [col+'_sum'+str(min_inc) for col in cols_to_add]] = \
                date_subDF[cols_to_add].rolling(min_inc, min_periods=1).sum().values

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
    unused_cols = ['Unnamed: 0', 'Product', 'Date', 'Minute', 'First', 'Last']
    used_cols = [col for col in all_minutesDF.columns if (col not in unused_cols)]

    # UNCOMMENT BELOW!!

    dailyDF.to_csv(config_dir+'Data/daily_summary2.csv') # for troubleshooting
    train_minutesDF.to_csv(config_dir+'Data/train_minutesDF.csv') # for troubleshooting

    if np.any(train_minutesDF[used_cols].isna()):
        for col in used_cols:
            if train_minutesDF[col].isna().sum() > 0:
                print(col, 'has', train_minutesDF[col].isna().sum(), 'na on dates', train_minutesDF.loc[train_minutesDF[col].isna()].Minute.dt.strftime('%Y-%m-%d').unique())
                print('dailyDF.head()', dailyDF.head())
                print('all_minutesDF.head()', all_minutesDF.head())


    assert(np.all(~train_minutesDF[used_cols].isna()))
    assert(np.all(~val_minutesDF[used_cols].isna()))
    used_cols = ['Minute']+list(used_cols) # We still need the minutes

    if saveProcessed:
        print('Saving processed data...')
        postprocessed_data_dir = config_dir+'Postprocessed_Data/'
        if not os.path.exists(postprocessed_data_dir): os.mkdir(postprocessed_data_dir)
        # [x] train_close_prices.csv and val_close_prices.csv
        train_close_pricesDF = pd.merge(train_minutesDF[['Minute']], all_minutesDF_orig[['Minute', 'C_B1', 'C_A1']], on='Minute', how='left')
        train_close_pricesDF = train_close_pricesDF.ffill()
        train_close_pricesDF.to_csv(config_dir+'Postprocessed_Data/train_close_prices.csv', index=False)
        val_close_pricesDF = pd.merge(val_minutesDF[['Minute']], all_minutesDF_orig[['Minute', 'C_B1', 'C_A1']], on='Minute', how='left')
        val_close_pricesDF = val_close_pricesDF.ffill()
        val_close_pricesDF.to_csv(config_dir+'Postprocessed_Data/val_close_prices.csv', index=False)

        train_minutesDF[used_cols].to_csv(postprocessed_data_dir+'train_minutesDF.csv', index=False)
        val_minutesDF[used_cols].to_csv(postprocessed_data_dir+'val_minutesDF.csv', index=False)

    print('Returning train with '+'{:,}'.format(len(train_minutesDF))+' rows and val with '+'{:,}'.format(len(val_minutesDF))+' rows.')

    return(train_minutesDF[used_cols], val_minutesDF[used_cols])


def loadAndProcessData(comb_row, config_dir, day_chg_incs, minute_incs,
                      minute_dir=tdm_dir+'Minute_Files/'):
    """
    1. Loads the minute files from the external hard drive
    2. Creates, saves, and returns all_minutesDF, sec_guideDF, and dailyDF
    """
    other_secs = comb_row[[col for col in comb_row.index if (col[:3]=='Sec' and col!='Sec1')]].values
    other_secs = [i for i in other_secs if type(i) == str]
    print('No pre-loaded data found. Loading data for '+comb_row.Sec1+' and '+','.join(other_secs))
    sec1_minuteDF = pd.read_csv(minute_dir+comb_row.Sec1+'.csv', parse_dates=['Date', 'Minute'])
    sec1_minuteDF.Date = sec1_minuteDF.Date.dt.date
    sec1_minuteDF = sec1_minuteDF.loc[(sec1_minuteDF.Date >= comb_row.TrainStartDate) & (sec1_minuteDF.Date <= comb_row.ValEndDate)].reset_index(drop=True)

    print('Loading minuteDF for '+comb_row.Sec2)
    other_secs_minuteDF = pd.read_csv(minute_dir+comb_row.Sec2+'.csv', parse_dates=['Date', 'Minute'])[['Product', 'Date', 'Minute', 'O_B', 'O_A', 'H_B', 'H_A', 'L_B', 'L_A', 'C_B', 'C_A', 'Count', 'B_TickImb', 'A_TickImb', 'M_TickImb']]
    for sec in other_secs[1:]:
        print('Loading minuteDF for '+sec)
        other_secs_minuteDF = other_secs_minuteDF.append(pd.read_csv(minute_dir+sec+'.csv', parse_dates=['Date', 'Minute'])[['Product', 'Date', 'Minute', 'O_B', 'O_A', 'H_B', 'H_A', 'L_B', 'L_A', 'C_B', 'C_A', 'Count', 'B_TickImb', 'A_TickImb', 'M_TickImb']], ignore_index=True)
    other_secs_minuteDF.Date = other_secs_minuteDF.Date.dt.date
    other_secs_minuteDF = other_secs_minuteDF.loc[(other_secs_minuteDF.Date >= comb_row.TrainStartDate) & (other_secs_minuteDF.Date <= comb_row.ValEndDate)].reset_index(drop=True)
    print('other_secs_minuteDF has '+str(len(other_secs_minuteDF))+' rows.')
    print('sec1_minuteDF has '+str(len(sec1_minuteDF))+' rows.')
    print("pd.read_csvs complete. Subsetting dates...")

    # [x] subset for dates
    dates_in_common = set(sec1_minuteDF.Date.unique())
    for sec in other_secs:
        print(sec)
        other_sec_dates = set(other_secs_minuteDF.loc[other_secs_minuteDF.Product == sec].Date.unique())
        print('removing', [str(d) for d in sorted(list(dates_in_common.difference(other_sec_dates)))])
        dates_in_common = dates_in_common.intersection(other_sec_dates)
        print(len(dates_in_common), 'dates_in_common')
    sec1_dates_to_remove = set(sec1_minuteDF.Date.unique()).difference(dates_in_common)
    print(str(len(dates_in_common))+' dates_in_common')

    if len(sec1_dates_to_remove) > 0:
        print('- removing '+str(len(sec1_dates_to_remove))+' dates from '+comb_row.Sec1)
        sec1_minuteDF = sec1_minuteDF.loc[sec1_minuteDF.Date.isin(dates_in_common)].reset_index(drop=True)
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
    for i in tqdm(range(len(dailyDF))):
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
    dailyDF.to_csv(config_dir+'Data/daily_summary.csv', index=False)
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
    for date in tqdm(dates_in_common):
        date = dailyDF.loc[i].Date
        if date in dates_in_common:
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
    all_minutesDF.to_csv(config_dir+'Data/all_minutes.csv', index=False)
    print('Save complete.')
    sec_guideDF = pd.DataFrame({'Sec': [comb_row.Sec1]+list(other_secs)})
    sec_guideDF.to_csv(config_dir+'Data/sec_guide.csv', index=False)
    return all_minutesDF, dailyDF, sec_guideDF

def processData(comb_row, config_dir, day_chg_incs, minute_incs):
    """
    1. Locates the Postprocessed_Data/*. If no Postprocessed_Data/* exists, then...
    2. Locates the Data/* and generates Postprocessed_Data/*. If no Data/* exists, then...
    3. Generate Data/* then generate Postprocessed_Data/*.
    """
    # [x] Locates the PostprocessedData/*
    # [x] check that it is in the correct format.
    sec_guide_fn = config_dir+'Data/sec_guide.csv'
    if os.path.exists(sec_guide_fn):
        sec_guideDF = pd.read_csv(sec_guide_fn)
        num_secs = len(sec_guideDF)
        postprocessed_data_dir = config_dir+'Postprocessed_Data/'
        if (os.path.exists(postprocessed_data_dir+'train_close_prices.csv') and
           os.path.exists(postprocessed_data_dir+'train_minutesDF.csv') and
           os.path.exists(postprocessed_data_dir+'val_close_prices.csv') and
           os.path.exists(postprocessed_data_dir+'val_minutesDF.csv')):
            print('All Postprocessed_Data files exist. Checking columns')

            cols_set_actual =  set(pd.read_csv(postprocessed_data_dir+'train_minutesDF.csv', nrows=0).columns.tolist())
            cols_set_expected = set(getExpectedPostprocessedCols(num_secs, day_chg_incs, minute_incs))

            assert cols_set_actual == cols_set_expected, 'Problem with the columns. \
            \nCols in Postprocessed actual but not in expected: '+str(cols_set_actual.difference(cols_set_expected))+\
            '\nCols in expected but not in Postprocessed actual: '+str(cols_set_expected.difference(cols_set_actual))

            print('Columns passed the check. Loading postprocessed data.')
            print('loading train_close_pricesDF')
            train_close_pricesDF = pd.read_csv(postprocessed_data_dir+'train_close_prices.csv')
            print('loading train_minutesDF')
            train_minutesDF = pd.read_csv(postprocessed_data_dir+'train_minutesDF.csv')
            print('loading val_close_pricesDF')
            val_close_pricesDF = pd.read_csv(postprocessed_data_dir+'val_close_prices.csv')
            print('loading val_minutesDF')
            val_minutesDF = pd.read_csv(postprocessed_data_dir+'val_minutesDF.csv')
            print('loading complete.')
            return train_close_pricesDF, train_minutesDF, val_close_pricesDF, val_minutesDF
        else:
            print('There are Postprocessed_Data/* missing. Here are the files there currently:')
            print(os.listdir(postprocessed_data_dir))
    else:
        print(sec_guide_fn, 'not found. Continuing...')

    print('\n')
    # [x] Locates or creates the Data/*
    data_dir = config_dir+'Data/'
    daily_fn = data_dir+'daily_summary.csv'
    all_minutes_fn = data_dir+'all_minutes.csv'
    sec_guide_fn = data_dir+'sec_guide.csv'

    all_minutesDF, dailyDF, sec_guideDF = 'not set yet', 'not set yet', 'not set yet'

    if not (os.path.exists(daily_fn) and os.path.exists(all_minutes_fn) and os.path.exists(sec_guide_fn)):
        print('Data/ has not been loaded and pre-processed yet. Doing so now.')
        all_minutesDF, dailyDF, sec_guideDF = loadAndProcessData(comb_row, config_dir, day_chg_incs, minute_incs)
    else:
        print('Data found! Loading data...')
        # [x] check the order of other_secs
        sec_guideDF = pd.read_csv(sec_guide_fn)
        assert(comb_row.Sec1 == sec_guideDF.iloc[0].Sec)
        other_secs_comb_row = comb_row[[col for col in comb_row.index if (col[:3]=='Sec' and col!='Sec1')]].values
        other_secs_comb_row = [i for i in other_secs_comb_row if type(i)==str]
        other_secs_sec_guide = sec_guideDF.iloc[1:].Sec.values
        assert(np.all(other_secs_comb_row == other_secs_sec_guide))
        # [x] check the dates
        dailyDF = pd.read_csv(daily_fn, parse_dates=['Date'])
        dailyDF.Date = dailyDF.Date.dt.date
        first_date = dailyDF.Date.iloc[0]
        last_date = dailyDF.Date.iloc[-1]
        assert abs((pd.to_datetime(first_date) - pd.to_datetime(comb_row.TrainStartDate)).days) < 20, str(first_date) +'   '+str(comb_row.TrainStartDate)  # may want to relax these
        assert abs((pd.to_datetime(last_date) - pd.to_datetime(comb_row.ValEndDate)).days) < 20, str(last_date) +'   '+str(comb_row.ValEndDate)
        all_minutesDF = pd.read_csv(all_minutes_fn, parse_dates=['Date', 'Minute'])
        all_minutesDF.Date = all_minutesDF.Date.dt.date
        print('Data/ load complete.')
    return postprocessData(all_minutesDF, dailyDF, sec_guideDF, config_dir, day_chg_incs, minute_incs)
