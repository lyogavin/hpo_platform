#!/usr/bin/env python
# coding: utf-8



import configparser
import copy

import torch
import collections
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
from train.data import postprocess_qa_predictions

class IncrementalAccumulateMeter(object):
    def __init__(self, config, previous_best=None):
        self.reset()
        self.best = previous_best
        self.last_best = None
        self.meter = AccumulateMeter()
        self.config = config
        self.res_list = []
    def reset(self, reset_best=False):
        self.res_list = []
        gc.collect()
        if reset_best:
            self.best = None
            self.last_best = None
    def get_features_count(self):
        return self.meter.get_features_count()
    def update(self, features, pred_starts, pred_ends, target_starts, target_ends):
        self.meter.update(features, pred_starts, pred_ends, target_starts, target_ends)

        res, _, _  = self.meter.get_metrics(None, self.config)
        self.res_list.append(res)

        self.meter.reset()
    def get_metrics(self, tokenzier, config):
        res = {}
        metrics = ['loss', 'jaccard']
        for m in metrics:
            res[m] = np.array([r[m] for r in self.res_list]).mean()
        is_best = False
        if self.best is None or res['jaccard'] > self.best:
            self.last_best = self.best
            self.best = res['jaccard']
            is_best = True
        return res, is_best, self.last_best




class AccumulateMeter(object):
    def __init__(self, previous_best=None):
        self.reset()
        self.best = previous_best
        self.last_best = None

    def reset(self, reset_best=False):
        self.features = []
        self.pred_starts = []
        self.pred_ends = []
        self.target_starts = []
        self.target_ends = []
        self.metrics = {}

        gc.collect()
        if reset_best:
            self.best = None
            self.last_best = None

    def update(self, features, pred_starts, pred_ends, target_starts, target_ends):
        self.pred_starts.extend(pred_starts.tolist())

        self.pred_ends.extend(pred_ends.tolist())
        self.target_starts.extend(target_starts.tolist())
        self.target_ends.extend(target_ends.tolist())

        #print(f"adding: {features.keys()}")
        to_extend = []
        #print(f"offset_mapping: {features['offset_mapping']}")
        for i in range(len(features['context'])):
            item = {}
            for k in features.keys():
                if 'probas' not in k:
                    to_copy = features[k].tolist()[i] if isinstance(features[k], torch.Tensor) else features[k][i]
                    item[k] = copy.deepcopy(to_copy) if not isinstance(to_copy,str) else to_copy
            to_extend.append(item)
        for x in to_extend:
            assert 'id' in x, f"{x.keys()} has to have id field"
        self.features.extend(to_extend)

    def get_features_count(self):
        return len(self.features)

    def get_metrics(self, tokenzier, config):
        if len(self.features) == 0:
            return {}, False, self.last_best

        try:
            res = get_metrics(self.features, tokenzier, self.pred_starts, self.pred_ends, self.target_starts, self.target_ends, config)
        except Exception as e:
            logging.info(f"exception getting metric for: {(self.features, self.pred_starts, self.pred_ends, self.target_starts, self.target_ends)}")
            raise e
        is_best = False
        if self.best is None or res['jaccard'] > self.best:
            self.last_best = self.best
            self.best = res['jaccard']
            is_best = True
        return res, is_best, self.last_best

def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    if len(a) + len(b) - len(c) == 0:
        return 0
    return float(len(c)) / (len(a) + len(b) - len(c))

def get_metrics(features, tokenzier, pred_starts, pred_ends, target_starts, target_ends, config):
    metrics = ['loss', 'jaccard']

    res_dict = {}

    for metric in metrics:
        if metric == 'loss':
            res = loss_fn((torch.tensor(pred_starts), torch.tensor(pred_ends)), (torch.tensor(target_starts), torch.tensor(target_ends))).item()
        elif metric == 'jaccard':
            #for context, pred_start, pred_end, target_start, target_end in \
            #    zip(contexts, pred_starts, pred_ends, target_starts, target_ends):
            #    logging.info(f"{pred_start}:{pred_end}, {target_start}:{target_end}")
            logging.debug(f"post process for {[x['start_position'] for x in features]}")
            logging.debug(f"post process for {[x['end_position'] for x in features]}")
            logging.debug(f"post process for {pred_starts}-{pred_ends}")
            text_predictions = postprocess_qa_predictions(tokenzier, features, pred_starts, pred_ends, use_char_model=config['USE_CHAR_MODEL'])
            logging.debug(f"text predictions: {text_predictions}")
            example_id_to_answers = {}
            for feat in features:
                example_id_to_answers[feat['id']] = feat['answer_text']
            logging.debug(f"text answers: {example_id_to_answers}")

            res = [jaccard(pred_text,
                           example_id_to_answers[example_id])
                   for example_id, pred_text in text_predictions.items()]
            res = np.array(res).mean()

            if config['DUMP_PRED']:
                import pickle
                dump_ts = int(time.time() / 60)
                with open(f'./dump_pred_{dump_ts}.pickle', 'wb') as f:
                    pickle.dump(text_predictions, f)
                    logging.info(f'pred dumped in: ./dump_pred_{dump_ts}.pickle')


        res_dict[metric] = res
    return res_dict

def pprint_metrics(res):
    return {k:f"{v:.4f}" for k,v in res.items()}


