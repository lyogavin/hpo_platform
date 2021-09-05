#!/usr/bin/env python
# coding: utf-8



import configparser
import torch
#import torchvision
import numpy as np
#import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
#from torchvision.transforms import ToTensor
#from torchvision.utils import make_grid
from torch.utils.data import random_split

from pathlib import Path
import pandas as pd
#import seaborn as sns
import gc
import time
from tqdm import tqdm
#import datatable as dt
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.model_selection import StratifiedKFold,KFold
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
#get_ipython().magic('matplotlib inline')

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
# only for debugging, this will block multiple GPU utilization
#os.environ["CUDA_LAUNCH_BLOCKING"]="1"   # see issue #152


# In[4]:


#!pip install transformers
import time


# set up logging:


import sys
sys.path.append('../')

from utils import utils

logging = utils.logging


# In[2]:


import numpy as np
from scipy.stats import norm
from scipy import stats
# referrencing: https://knowledge-repo.d.musta.ch/post/projects/datau332_recalculating_erf_metrics.kp


                
import math



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
from train.loss import loss_fn


class AccumulateMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.contexts = []

        self.pred_starts = []
        self.pred_ends = []
        self.target_starts = []
        self.target_ends = []
        self.metrics = {}
        self.best = None

    def update(self, contexts, pred_starts, pred_ends, target_starts, target_ends):
        self.pred_starts.extend(pred_starts.item())
        self.pred_ends.extend(pred_ends.item())
        self.target_starts.extend(target_starts.item())
        self.target_ends.extend(target_ends.item())
        self.contexts.extend(contexts.item())

    def get_metrics(self):
        res = get_metrics(self.contexts, self.pred_starts, self.pred_ends, self.target_starts, self.target_ends)
        is_best = False
        if self.best is None or res['jaccard'] > self.best:
            last_best = self.best
            self.best = res['jaccard']
            is_best = True
        return res, is_best, last_best

def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def get_metrics(contexts, pred_starts, pred_ends, target_starts, target_ends):
    metrics = ['loss', 'jaccard']

    res_dict = {}

    for metric in metrics:
        if metric == 'loss':
            res = loss_fn((pred_starts, pred_ends), (target_starts, target_ends))
        elif metric == 'jaccard':
            res = [jaccard(context[pred_start:pred_end], context[target_start:target_end])
                   for context, pred_start, pred_end, target_start, target_end in
                   zip(contexts, pred_starts, pred_ends, target_starts, target_ends)]
            res = np.array(res).mean()

        res_dict[metric] = res
    return res_dict