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



class Lamb(Optimizer):
    # Reference code: https://github.com/cybertronai/pytorch-lamb

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0,
        clamp_value: float = 10,
        adam: bool = False,
        debias: bool = False,
    ):
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        if clamp_value < 0.0:
            raise ValueError('Invalid clamp value: {}'.format(clamp_value))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.clamp_value = clamp_value
        self.adam = adam
        self.debias = debias

        super(Lamb, self).__init__(params, defaults)

    def step(self, closure = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    msg = (
                        'Lamb does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                    raise RuntimeError(msg)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Paper v3 does not use debiasing.
                if self.debias:
                    bias_correction = math.sqrt(1 - beta2 ** state['step'])
                    bias_correction /= 1 - beta1 ** state['step']
                else:
                    bias_correction = 1

                # Apply bias to lr to avoid broadcast.
                step_size = group['lr'] * bias_correction

                weight_norm = torch.norm(p.data).clamp(0, self.clamp_value)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group['weight_decay'])

                adam_norm = torch.norm(adam_step)
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(adam_step, alpha=-step_size * trust_ratio)

        return loss


# In[18]:


def get_optimizer_params(model, config):
    # differential learning rate and weight decay
    param_optimizer = list(model.named_parameters())
    learning_rate = config['LR']

    no_decay = ['bias', 'gamma', 'beta']
    group1=['layer.0.','layer.1.','layer.2.','layer.3.']
    group2=['layer.4.','layer.5.','layer.6.','layer.7.']    
    group3=['layer.8.','layer.9.','layer.10.','layer.11.']
    group_all=['layer.0.','layer.1.','layer.2.','layer.3.','layer.4.','layer.5.','layer.6.','layer.7.','layer.8.','layer.9.','layer.10.','layer.11.']
    optimizer_parameters = [
        {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay_rate': 0.01},
        {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay_rate': 0.01, 'lr': learning_rate/2.6},
        {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay_rate': 0.01, 'lr': learning_rate},
        {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay_rate': 0.01, 'lr': learning_rate*2.6},
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay_rate': 0.0},
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay_rate': 0.0, 'lr': learning_rate/2.6},
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay_rate': 0.0, 'lr': learning_rate},
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay_rate': 0.0, 'lr': learning_rate*2.6},
        {'params': [p for n, p in model.named_parameters() if "roberta" not in n], 'lr':learning_rate*10, "momentum" : 0.99},
    ]
    return optimizer_parameters


# In[19]:

def make_layered_optimizer(model, config):
    #from torch.optim import AdamW
    named_parameters = list(model.named_parameters())   
    
    roberta_max_layers = {
        'roberta-base':197,
        'albert-xxlarge-v2':25,
        'microsoft/deberta-large':388,
        'roberta-large':391,
    }
    roberta_mid_layers = {
        'roberta-base':69,
        'albert-xxlarge-v2':7,
        'microsoft/deberta-large':131,
        'roberta-large':133
    }
    roberta_late_layers = {
        'roberta-base':133,
        'albert-xxlarge-v2':23,
        'microsoft/deberta-large':259,
        'roberta-large':261,
    }
    attention_min_layers = {
        'roberta-base':199,
        'albert-xxlarge-v2':25,
        'microsoft/deberta-large':388,
        'roberta-large':391
    }
    attention_max_layers = {
        'roberta-base':203,
        'albert-xxlarge-v2':29,
        'microsoft/deberta-large':392,
        'roberta-large':395
    }
    regressor_min_layers = {
        'roberta-base':203,
        'albert-xxlarge-v2':29,
        'microsoft/deberta-large':392,
        'roberta-large':395
    }
    
    model_bert_path = config['BERT_PATH']
    if model_bert_path not in config['LAYERED_OPT_ENABLED'].split(','):
        model_bert_path = 'roberta-base'
        
    logging.info(f"layered opt model type to use: {model_bert_path}")
    
    roberta_parameters = named_parameters[:roberta_max_layers[model_bert_path]]    
    attention_parameters = named_parameters[attention_min_layers[model_bert_path]:attention_max_layers[model_bert_path]]
    regressor_parameters = named_parameters[regressor_min_layers[model_bert_path]:]
        
    attention_group = [params for (name, params) in attention_parameters]
    regressor_group = [params for (name, params) in regressor_parameters]

    parameters = []
    
    
    to_append = {}
    if config['LAYERED_OPT_DEFAULT_WEIGHT_DECAY'] is not None:
        to_append['weight_decay'] = config['LAYERED_OPT_DEFAULT_WEIGHT_DECAY']
    if config['LAYERED_OPT_DEFAULT_LR'] is not None:
        to_append['lr'] = config['LAYERED_OPT_DEFAULT_LR']
        
    parameters.append({**{"params": attention_group},**to_append})
    parameters.append({**{"params": regressor_group},**to_append})
    
    logging.info(f"layered opt parameters used for attention and regressor layers: {to_append}")


    learning_rate = config['LR']

    for layer_num, (name, params) in enumerate(roberta_parameters):
        weight_decay = 0.0 if "bias" in name else 0.01

        lr = learning_rate #2e-5

        if layer_num >= roberta_mid_layers[model_bert_path]:        
            lr = learning_rate * 2.5 #5e-5

        if layer_num >= roberta_late_layers[model_bert_path]:
            lr = learning_rate * 5.0 #1e-4

        parameters.append({"params": params,
                           "weight_decay": weight_decay,
                           "lr": lr})
        
        to_disp = {"weight_decay": weight_decay,
                           "lr": lr}
    logging.info(f"layered opt parameters used for late roberta layers: {to_disp}")

    return AdamW(parameters)

def make_optimizer(model, config):
    optimizer_grouped_parameters = get_optimizer_params(model, config)
    optimizer_name=config['OPTIMIZER_NAME']
    
    kwargs = {
            'lr':config['LR'],
            'weight_decay': config['WEIGHT_DECAY']
    }
    
    if config['ADAMW_BETAS'] is not None:
        kwargs['betas'] = config['ADAMW_BETAS']
    if config['ADAMW_EPS'] is not None:
        kwargs['eps'] = config['ADAMW_EPS']
    
    if optimizer_name == "LAMB":
        optimizer = Lamb(optimizer_grouped_parameters, **kwargs)
        return optimizer
    elif optimizer_name == "Adam":
        from torch.optim import Adam
        optimizer = Adam(optimizer_grouped_parameters, **kwargs)
        return optimizer
    elif optimizer_name == "AdamW":
        #from torch.optim import AdamW
        optimizer = AdamW(optimizer_grouped_parameters, **kwargs)
        return optimizer
    elif optimizer_name == "LayeredOptimizer":
        return make_layered_optimizer(model, config)
    else:
        raise Exception('Unknown optimizer: {}'.format(optimizer_name))

def make_scheduler(optimizer, train_loader, config):
    decay_name=config['DECAY_NAME'] #'cosine_warmup',

    #t_max=config['EPOCHS']
    

    grad_accu_factor = config['GRAD_ACCU_STEPS']

    t_max = int(len(train_loader)  *
                config['EPOCHS'] / grad_accu_factor)

            
    if isinstance(config['WARMUP_STEPS_RATIO'], float):
        warmup_steps = config['WARMUP_STEPS_RATIO'] * t_max
    elif isinstance(config['WARMUP_STEPS_RATIO'], int):
        warmup_steps = config['WARMUP_STEPS_RATIO']
    else:
        warmup_steps = 0
        
    print(f"using warmup steps: {warmup_steps}")
    logging.info(f"using warmup steps: {warmup_steps}")
        
    if decay_name == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[30, 60, 90],
            gamma=0.1
        )
    elif decay_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max
        )
    elif decay_name == "cosine_warmup":
        func_args = {
            'optimizer':optimizer,
            'num_warmup_steps':warmup_steps,
            'num_training_steps':t_max
        }
        if config['NUM_CYCLES'] is not None:
            func_args['num_cycles'] = config['NUM_CYCLES']
        scheduler = get_cosine_schedule_with_warmup(**func_args
        )
    elif decay_name == "cosine_warmup_hard_restart":
        
        func_args = {
            'optimizer':optimizer,
            'num_warmup_steps':warmup_steps,
            'num_training_steps':t_max
        }
        if config['NUM_CYCLES'] is not None:
            func_args['num_cycles'] = config['NUM_CYCLES']
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            **func_args
        )
    elif decay_name == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps
        )
    elif decay_name == "polynomial":
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=t_max,
            lr_end=config['POLY_DECAY_LR_END'] *
                   config['LR']
        )
    elif decay_name == "linear":
        scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps = warmup_steps,
                num_training_steps=t_max,
            )
    
    else:
        raise Exception('Unknown lr scheduler: {}'.format(decay_name))    
    return scheduler    

