import pytz, os
import pandas as pd
import numpy as np


def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    # except ImportError:
    except: #I think it's ok to treat any error as not in a notebook
        return False
    return True

if in_notebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

def utcToChi(utc_dt):
    chi_tz = pytz.timezone('America/Chicago')
    chi_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(chi_tz)
    return chi_tz.normalize(chi_dt)

def getChiTimeNow():
    utc_dt = pd.datetime.utcnow()
    return(utcToChi(utc_dt))

def readCSV(filename, nrows=None):
    """A custom pandas csv reader which handles dates appropriately by default"""
    cols = pd.read_csv(filename, nrows=1).columns
    # date_cols = [c for c in ['Date', 'date'] if c in cols]
    date_cols = [c for c in cols if 'date' in c.lower()]
    timestep_cols = [c for c in ['Minute', 'Minutes', 'minute', 'minutes',
                                 'Second', 'Seconds', 'second', 'seconds',
                                 'Time', 'time', 'Timestep', 'timestep'] if c in cols]
    DF = pd.read_csv(filename, nrows=nrows, parse_dates=date_cols+timestep_cols)
    for c in date_cols: DF[c] = DF[c].dt.date
    return DF

def autoChooseSecurity(omni_dir = os.getcwd()):
    """For now, let's just figure out if it's being worked on in another job, and how many models have been tried"""
    print('in autoChooseSecurity with omni_dir =', omni_dir)
    securities = os.listdir(omni_dir+'/Primary_Assets')
    model_countDF = pd.DataFrame({'Sec': securities, 'ModelCount': 0})
    for sec in securities:
        sec_dir = omni_dir+'/Primary_Assets/'+sec+'/'
        configs = [d for d in os.listdir(sec_dir) if d[:6]=='Config']
        if len(configs) == 0:
            print('No configs for', sec)
        else:
            config_nums = [int(cfg.split('Config')[1]) for cfg in configs]
            max_config = str(max(config_nums))
            config_dir = sec_dir + 'Config' + max_config + '/'
            all_progress_summary_fn = config_dir + 'Deep_Models/all_progress_summary.csv'
            if not os.path.exists(all_progress_summary_fn):
                print('\nall_progress_summary_fn does not exist: \n', all_progress_summary_fn)
            else:
                all_progress_summaryDF = pd.read_csv(all_progress_summary_fn)
                model_count = len(all_progress_summaryDF.Model.unique())
                model_countDF.loc[model_countDF.Sec == sec, 'ModelCount'] = model_count

    work_log_fn = omni_dir + 'work_log.csv'
    time_now = pd.datetime.now()
    num_hours_wait = 6

    work_logDF = pd.DataFrame(columns=['Sec', 'ChosenTime'])
    if not os.path.exists(work_log_fn):
        print('work log doesnt exist. Creating a new one:', work_log_fn)
    else:
        work_logDF = pd.read_csv(work_log_fn, parse_dates=['ChosenTime'])
        # [x] remove from model_countDF any security that's been worked on less than num_hours_wait ago
        last_chosenDF = work_logDF.groupby('Sec').agg({'ChosenTime': 'max'})
        last_chosenDF['Sec'] = last_chosenDF.index
    #     secs_chosen_recently = last_chosenDF.Sec.loc[float((time_now - last_chosenDF.ChosenTime).seconds)/60/60 < num_hours_wait]
        secs_chosen_recently = last_chosenDF.Sec.loc[(time_now - last_chosenDF.ChosenTime).astype('timedelta64[h]') < num_hours_wait]
        secs_chosen_recently = secs_chosen_recently.values
        print('secs_chosen_recently:', secs_chosen_recently)
        model_countDF = model_countDF.loc[~model_countDF.Sec.isin(secs_chosen_recently)]
    assert len(model_countDF) > 0, "We have no securities that have not been tried recently"
    # Randomly choose a weighted choice in the bottom 30% of ModelCounts.
    # This is to prevent a situation where one security is throwing errors
    # for some reason and we keep trying it unsuccessfully.
    threshold = model_countDF.ModelCount.quantile(.3)
    model_count_subDF = model_countDF.loc[model_countDF.ModelCount <= threshold]
    model_count_subDF['Weights'] = (threshold - model_countDF.ModelCount)+.1 #adds .1 because if we didn't then some weights would be 0
    model_count_subDF.Weights /= model_count_subDF.Weights.sum()
    chosen_security = np.random.choice(model_count_subDF.Sec.values, p=model_count_subDF.Weights.values)
    print('\n\n\n', '='*50, '\nchosen_security:', chosen_security, '\n', '='*50, '\n\n\n')
    return chosen_security
