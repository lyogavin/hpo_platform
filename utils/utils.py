#!/usr/bin/env python

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data import random_split

from pathlib import Path
import pandas as pd
import gc
import time
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.model_selection import StratifiedKFold,KFold
from scipy.stats import spearmanr


import os
import time
import random
import gc
import subprocess

from colorama import Fore, Back, Style
red = Fore.RED
grn = Fore.GREEN
blu = Fore.BLUE
ylw = Fore.YELLOW
wht = Fore.WHITE
blk = Fore.BLACK
bred = Back.RED
bgrn = Back.GREEN
bblu = Back.BLUE
bylw = Back.YELLOW
bwht = Back.WHITE
#rst = Style.RESET

import plotly.express as ex
import plotly.graph_objs as go
import plotly.figure_factory as ff

from filelock import FileLock

import os
import time
import logging, sys

logging_file_path = f"crp/training_log_{run_ts}.log"

if os.path.exists("crp"):
    handlers=[
        logging.FileHandler(logging_file_path),
        #logging.StreamHandler(sys.stdout)
    ]
else:
    handlers=[
        #logging.FileHandler(logging_file_path),
        logging.StreamHandler(sys.stdout)
    ]
    

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=handlers
)


# In[2]:


import numpy as np
from scipy.stats import norm
from scipy import stats
# referrencing: https://knowledge-repo.d.musta.ch/post/projects/datau332_recalculating_erf_metrics.kp

import math

def bootstrap_rmse(pred, target, sample_size=0.5, sample_count=100, alpha=0.9, squared=False, low_high=False):
    values = []
    se = torch.nn.MSELoss(reduction='none')(torch.tensor(pred), torch.tensor(target)).detach().cpu().numpy()
    #print(f"bootstrap_rmse input: se:{se}\n, se shape:{se.shape}\n")
    
    
    rmse = se.mean()
    if not squared:
        rmse = math.sqrt(rmse)
        
    if not low_high:
        return (rmse, 0, 0)
        
    to_sample = pd.Series(se)
    
    for i in range(sample_count):
        sampled = to_sample.sample(int(sample_size * len(pred)))
        to_append = sampled.mean()
        
        if not squared:
            to_append = math.sqrt(to_append)
        
        values.append(to_append)
        
    # confidence intervals
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(values, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(values, p))
    
    return (rmse, lower,upper)

# inputs are 2 lists where each record is zero or one in each sample entering the experiment:

def calculate_p_value_mde(control_points_or_not_list, treatment_point_or_not_list):
    # 2. Sums and means for numerator and denominator
    
    assignments_control = len(control_points_or_not_list)
    assignments_treatment = len(treatment_point_or_not_list)
    
    if assignments_control == 0 or assignments_treatment == 0:
        return 0., 0., 0.

    sum_control_numerator = control_points_or_not_list.sum()
    sum_treatment_numerator = treatment_point_or_not_list.sum()
    
    sum_control_denominator = assignments_control # If ratio metric, we would sum the denominator values instead
    sum_treatment_denominator = assignments_treatment # If ratio metric, we would sum the denominator values instead

    mean_control_numerator = sum_control_numerator / assignments_control
    mean_control_denominator = sum_control_denominator / assignments_control
    mean_treatment_numerator = sum_treatment_numerator / assignments_treatment
    mean_treatment_denominator = sum_treatment_denominator / assignments_treatment
    
    # 3. For sample second moments 
    
    sum_control_numerator2 = np.sum(control_points_or_not_list**2)
    sum_control_denominator2 = assignments_control # If ratio metric, we would sum the denominator^2 values instead
    sum_treatment_numerator2 = np.sum(treatment_point_or_not_list**2)
    sum_treatment_denominator2 = assignments_treatment # If ratio metric, we would sum the denominator^2 values instead
    
    # 4. Variances of numerator and denominator
    
    variance_control_numerator = sum_control_numerator2 / assignments_control - (sum_control_numerator / assignments_control)**2
    variance_control_denominator = sum_control_denominator2 / assignments_control - (sum_control_denominator / assignments_control)**2
    variance_treatment_numerator = sum_treatment_numerator2 / assignments_treatment - (sum_treatment_numerator / assignments_treatment)**2
    variance_treatment_denominator = sum_treatment_denominator2 / assignments_treatment - (sum_treatment_denominator / assignments_treatment)**2
    
    
    
    # 5. Cross moments (more info)
    
    cross_moment_control = mean_control_numerator * mean_control_denominator  # adjust this if getting a ratio metric to cross-sum obtained directly from query
    cross_moment_treatment = mean_treatment_numerator * mean_treatment_denominator  # see above
    
    # 6. Covariance
    
    co_var_control = cross_moment_control - mean_control_numerator * mean_control_denominator # 0 bc not ratio metric
    co_var_treatment = cross_moment_treatment - mean_treatment_numerator * mean_treatment_denominator # 0 bc not ratio metric
    
    
    # 7. Variances of ratio metrics
    
    variance_treatment =         mean_treatment_denominator**-2         * (
            variance_treatment_numerator \
            + variance_treatment_denominator * (mean_treatment_numerator / mean_treatment_denominator)**2 \
            - (2 * co_var_treatment * (mean_treatment_numerator /  mean_treatment_denominator))
        )

    variance_control =         mean_control_denominator**-2         * (
            variance_control_numerator \
            + variance_control_denominator * (mean_control_numerator / mean_control_denominator)**2 \
            - (2 * co_var_control * (mean_control_numerator /  mean_control_denominator))
        )

    
    # 8. Means
    mean_control = mean_control_numerator / mean_control_denominator
    mean_treatment = mean_treatment_numerator / mean_treatment_denominator
    
    if mean_control == 0.:
        percent_change = 0.
    else:
        percent_change = (mean_treatment - mean_control) / mean_control
    
    # 9. Standard error, Z-statistic, and p-value
    
    stderr = ((variance_treatment/assignments_treatment) + (variance_control/assignments_control))**0.5
    zstat = (mean_treatment - mean_control) / stderr if stderr != 0 else 0
    pvalue = 2 * (1 - norm.cdf(abs(zstat)))
    ci_95 = stderr * 1.959967124 / mean_control  if mean_control != 0 else 0
    
    # 10. Comparing against ground truth
    
    if False:
        print("sum_control_numerator is {}".format(sum_control_numerator))
        print("sum_control_denominator is {}".format(sum_control_denominator))
        print("sum_treatment_numerator is {}".format(sum_treatment_numerator))
        print("sum_treatment_denominator is {}".format(sum_treatment_denominator))
        print("mean_treatment is {}".format(mean_treatment))
        print("mean_control is {}".format(mean_control))
        print("percent_change is {}".format(percent_change))
        print("pvalue is {}".format(pvalue))
        print("stderr is {}".format(stderr))
        print("ci_95 is {}".format(ci_95))
    
    
    
    # Power:
    # let's work from values we would see on the ERF UI; hint, in terms of 
    # scipy stats functions, you might need:
    # norm.isf - returns z critical value on right tail
    # norm.ppf - returns z critical value on  left tail
    # norm.sf  - returns area under curve right of a z critical value
    # norm.cdf - returns area under curve  left of a z critical value

    detectable_lift = 0.73            # MDE on ERF UI
    base_rate       = mean_control    # 45 / 17986
    standard_error  = stderr          # calculated earlier!
    alpha           = 0.05 / 2        # two-tailed
    
    power    = 0
    delta    = detectable_lift * base_rate

    Zcrit_r  = stats.norm.isf(alpha)                             # +1.96, as expected
    power   += stats.norm.sf( Zcrit_r - delta / standard_error)  if standard_error != 0 else 0 # norm.sf is area to right tail

    Zcrit_l  = stats.norm.ppf(alpha)                             # -1.96, as expected
    power   += stats.norm.cdf(Zcrit_l - delta / standard_error)  if standard_error != 0 else 0 # norm.cdf is area to left tail

    #print("")
    #print("Power corresponding to a 73% MDE is: {:0.2f}".format(power)) # shockingly close eh...

    # MDE
    
    MDD = 2.80158178701

    mde = MDD * standard_error / mean_control if mean_control != 0 else 0
    #print("")
    #print("MDE is {:0.2f}".format(mde)) # shockingly close eh?! 
    
    return pvalue, mde, percent_change


# In[6]:


from transformers import AutoTokenizer,AutoModelForSequenceClassification,BertModel,DebertaTokenizer,BertTokenizer
from transformers import InputExample, InputFeatures
from transformers.file_utils import WEIGHTS_NAME
from transformers import RobertaConfig, RobertaModel
from torch.optim.optimizer import Optimizer
from transformers import (
    get_cosine_schedule_with_warmup, 
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup
)
import os


# In[7]:


def seed_everything(seed=43):
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True
    #config['SEED'] = seed


import copy

def diff_dict(dict1, dict2, to_exclude = keys_to_exclude_in_hash):
    dict1 = copy.copy(dict1)
    dict2 = copy.copy(dict2)
    for k in to_exclude:
        if k in dict1:
            del dict1[k]
        if k in dict2:
            del dict2[k]
    try:
        set1 = set([(k,tuple(v)) if isinstance(v, list) else (k,v) for k,v in dict1.items()])
    except TypeError as te:
        print(te)
        raise te
    try:
        set2 = set([(k,tuple(v)) if isinstance(v, list) else (k,v) for k,v in dict2.items()])
    except TypeError as te:
        print(te)
        raise te
    return {k:dict1[k] for k in dict(set1 ^ set2).keys()}

