import os
import pandas as pd
import numpy as np

def getCombDF(regression_results_fn, data_summary_fn):
    print('Getting combDF using ', regression_results_fn, 'and', data_summary_fn)
    # Determine which combinations of data to try

    # load prepare regressionDF
    regression_cols = ['Sec1', 'Sec2', 'R2', 'p_value', 'NumDatapoints', 'StartDate', 'EndDate']

    regressionDF = pd.read_csv(regression_results_fn)
    data_summaryDF = pd.read_csv(data_summary_fn)
    for col in regressionDF.columns:
        if 'Unnamed' in col:
            regressionDF.drop(columns=[col], inplace=True)

    assert(set(regression_cols) == set(regressionDF.columns))
    regressionDF = regressionDF[regression_cols]

    print(str(len(regressionDF.loc[regressionDF.R2 == 'not found']))+' rows have R2 not found:\n')
    print(regressionDF.loc[regressionDF.R2 == 'not found'][['Sec1', 'Sec2']])

    regressionDF = regressionDF.loc[regressionDF.R2 != 'not found'].reset_index(drop=True)
    regressionDF.R2 = regressionDF.R2.astype(float)
    # regressionDF




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




    # construct combDF
    comb_cols = ['Sec1']+['Sec'+str(i) for i in range(2,(max_sec_bundle+2))] \
        +['R2_'+str(i) for i in range(2,(max_sec_bundle+2))] \
        +['p_value'+str(i) for i in range(2,(max_sec_bundle+2))] \
        +['StartDate'+str(i) for i in range(2,(max_sec_bundle+2))] \
        +['EndDate'+str(i) for i in range(2,(max_sec_bundle+2))] \
        +['NumDatapoints'+str(i) for i in range(2,(max_sec_bundle+2))]

    combDF = pd.DataFrame(columns = comb_cols)

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
    # combDF



    ##### Determine which sets of dates to try
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

    for i in range(1, max_sec_bundle+2):
        combDF['TotalStartDate'+str(i)] = np.nan
        combDF['TotalEndDate'+str(i)] = np.nan

    for i in range(len(combDF)):
        #get total Start and End dates
        for j in range(1, max_sec_bundle+2):
            if type(combDF['Sec'+str(j)][i]) == str:
                combDF.loc[combDF['Sec'+str(j)] == combDF['Sec'+str(j)][i], 'TotalStartDate'+str(j)] = data_summaryDF.loc[data_summaryDF.Name == combDF['Sec'+str(j)][i], 'StartDate'].values[0]
                combDF.loc[combDF['Sec'+str(j)] == combDF['Sec'+str(j)][i], 'TotalEndDate'+str(j)] = data_summaryDF.loc[data_summaryDF.Name == combDF['Sec'+str(j)][i], 'EndDate'].values[0]




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
    # combDF


    # convert the date columns into actual dates
    date_cols = [col for col in combDF.columns if 'Date' in col]
    for col in date_cols:
        combDF[col] = pd.to_datetime(combDF[col]).dt.date



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
            print('num_days < min_num_days_total! Skipping row for '+combDF.Sec1.loc[i])
        else:
            combDF.loc[i, 'TrainStartDate'] = combDF.loc[i, 'LastStartDate']
            combDF.loc[i, 'ValEndDate'] = combDF.loc[i, 'FirstEndDate'] - pd.Timedelta(days=num_secondary_val_days)
            combDF.loc[i, 'ValStartDate'] = combDF.loc[i, 'TrainStartDate'] + pd.Timedelta(days=int(num_days*train_pct))
    # combDF
    return combDF
