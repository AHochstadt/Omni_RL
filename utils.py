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

def utcToChi(utc_dt):
    chi_tz = pytz.timezone('America/Chicago')
    chi_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(chi_tz)
    return chi_tz.normalize(chi_dt)

def getChiTimeNow():
    utc_dt = pd.datetime.utcnow()
    return(utcToChi(utc_dt))
