import numpy as np
import pandas as pd
    
def bootstrap(data, outcome, N = 10000):
    """
    Boostrapping assuming all successes are similar and all failures are similar
    Parameters
    ----------
    data : pd.DataFrame
        Data being bootstrapped.
    outcome : dict['str'->'str']
        Maps column of data that is our outcome to the level that is a successful outcome (a 1)
        Assumes anything else is a failure (a 0)
    N : int, optional
        Number of observations for each group after bootstrap. The default is 10000.

    Returns
    -------
    pd.DataFrame bootstrap sample containing n rows of success and n rows of failure
    """
    key = list(outcome.keys())[0]
    value = list(outcome.values())[0]
    successes = data[data[key] == value]
    failures = data[data[key] != value]
    temp_succ = pd.DataFrame(columns = data.columns)
    while successes.shape[0] + temp_succ.shape[0] < N:
        temp_Ser = pd.Series(index = data.columns)
        for x in data.columns:
            if x in ['Pushup', 'Situp', 'Chinup', 'Run']:
                if temp_Ser['SEX_ENC']:
                    temp_dat = successes[successes['SEX_ENC'] == 1]
                    mean = temp_dat[x].mean()
                    sd = temp_dat[x].std()
                    temp_Ser[x] = np.random.normal(mean, sd)
                else:
                    temp_dat = successes[successes['SEX_ENC'] == 0]
                    mean = temp_dat[x].mean()
                    sd = temp_dat[x].std()
                    temp_Ser[x] = np.random.normal(mean, sd)
            elif x != 'SEX_ENC' and isinstance(data[x][0], np.float64):
                mean = successes[x].mean()
                sd = successes[x].std()
                temp_Ser[x] = np.random.normal(mean, sd)
            else:
                temp_Ser[x] = np.random.choice(successes[x])
        temp_succ = temp_succ.append(temp_Ser, ignore_index = True)
    successes = successes.append(temp_succ, ignore_index = True)
    temp_fail = pd.DataFrame(columns = data.columns)
    while failures.shape[0] + temp_fail.shape[0] < N:
        temp_Ser = pd.Series(index = data.columns)
        for x in data.columns:
            if x in ['Pushup', 'Situp', 'Chinup', 'Run']:
                if temp_Ser['SEX_ENC']:
                    temp_dat = failures[failures['SEX_ENC'] == 1]
                    mean = temp_dat[x].mean()
                    sd = temp_dat[x].std()
                    temp_Ser[x] = np.random.normal(mean, sd)
                else:
                    temp_dat = failures[failures['SEX_ENC'] == 0]
                    mean = temp_dat[x].mean()
                    sd = temp_dat[x].std()
                    temp_Ser[x] = np.random.normal(mean, sd)
            elif x != 'SEX_ENC' and isinstance(data[x][0], np.float64):
                mean = failures[x].mean()
                sd = failures[x].std()
                temp_Ser[x] = np.random.normal(mean, sd)
            else:
                temp_Ser[x] = np.random.choice(failures[x])
        temp_fail = temp_fail.append(temp_Ser, ignore_index = True)
    failures = failures.append(temp_fail, ignore_index = True)
    successes = successes.append(failures, ignore_index = True)
    return successes.sample(frac = 1)