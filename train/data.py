#!/usr/bin/env python
# coding: utf-8



import configparser
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
get_ipython().magic('matplotlib inline')

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

# In[5]:


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



class CommonLitDataset(nn.Module):
    def __init__(self, data, tokenizer, config, reweighting=False):
        self.config = config
        max_len = config.getint(configparser.DEFAULTSECT, 'MAX_LEN', fallback=256)
        self.excerpt = data['excerpt'].to_numpy()
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.targets = data['target'] if 'target' in data.columns else None

        self.weights = None
        if reweighting and 'reweighting' in data.columns:
            self.weights = data['reweighting']
            
        if config.getboolean(configparser.DEFAULTSECT,'TOKENIZE_ALL', fallback=False):
            self.encoded = self.tokenizer(data['excerpt'].to_list(),
                            max_length=max_len,
                            padding=self.config.get(configparser.DEFAULTSECT, 'TOKENIZER_PADDING', fallback=None),
                            truncation=True)

            
        if config.getboolean('runtime','DEBUG_PRINT', fallback=False):
            print(data.head())
        
    def __len__(self):
        return len(self.excerpt)
    
    def __getitem__(self,item):
        excerpt = self.excerpt[item]
        if self.config.getboolean(configparser.DEFAULTSECT, 'REMOVE_NEWLINE', fallback=False):
            excerpt = excerpt.replace('\n', '')
        
        if config.getboolean(configparser.DEFAULTSECT,'TOKENIZE_ALL', fallback=False):
            inputs = {'input_ids':torch.tensor(self.encoded['input_ids'][item]),
                      'attention_mask':torch.tensor(self.encoded['attention_mask'][item])
                     }
        else:
            inputs = self.tokenizer(excerpt,
                                max_length=config.getint(configparser.DEFAULTSECT, 'MAX_LEN', fallback=256),
                                padding=self.config.get(configparser.DEFAULTSECT, 'TOKENIZER_PADDING', fallback=None),
                                truncation=True,
                                return_tensors='pt')
        if self.targets is not None:
            target = torch.tensor(self.targets[item], dtype=torch.float if self.loss_type != 'multi-class' else torch.long)  
            if self.weights is not None:
                            
                weight = torch.tensor(self.weights[item], dtype=torch.float) 

                return inputs,target, weight
            else:
                                
                if self.config.getboolean('runtime','DEBUG_PRINT', fallback=False):
                    return inputs,target, excerpt
                else:
                    return inputs,target

                
        else:
            return inputs
        


class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True, random_state=42):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle, random_state=random_state)
        self.X = torch.randn(len(y),1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return len(self.y)

