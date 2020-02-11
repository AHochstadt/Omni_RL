import pytz
import pandas as pd

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
