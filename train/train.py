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
from distutils import file_util, dir_util
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
import git

#!pip install transformers
import time


TRAIN_MODE = True
TEST_MODE = False


# In[5]:


# set up logging:


import sys
sys.path.append('../')
sys.path.append('./')

from utils.utils import *
from train.loss import *
from train.data import *
from train.optimizer import *
from model.model import *
from train.metric import *
from model import model
from exp_record_store.exp_record import ExpRecord
from utils.utils import logging
from utils.timer import Timer
#logging = utils.logging


# In[2]:


import numpy as np
from scipy.stats import norm
from scipy import stats
# referrencing: https://knowledge-repo.d.musta.ch/post/projects/datau332_recalculating_erf_metrics.kp



import_file_name = 'model'

model_import = __import__(import_file_name)
model_import = getattr(model_import, 'model')


                
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


# In[7]:

def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (:obj:`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model



DEBUG_PRINT = False

from pathlib import Path
import distutils

def save_run(original_model, paralelled_model, tokenizer, config, saving_dir, fold):

    with Timer("training saving") as saving_timer:

        original_model.save_pretrained(paralelled_model, f'{saving_dir}/model_{fold}')
        save_training_config(config, f'{saving_dir}/model_{fold}')
        tokenizer.save_pretrained(f'{saving_dir}/model_{fold}/tokenizer')

        # save config another copy in tokenizer
        original_model.roberta.config.save_pretrained(f'{saving_dir}/model_{fold}/tokenizer')

        # save source files...
        path = Path(os.path.dirname(os.path.abspath(__file__)))
        dir_util.copy_tree(path.parent.absolute(), f'{saving_dir}/src/')

        logging.info(f"saved in {saving_dir}/model_{fold}")

        #import shutil
        #shutil.make_archive(f"{saving_dir}", 'zip', f"{saving_dir}")

    logging.info(saving_timer.get_total_secs_str())



def train_fn(data_loader, valid_loader,
             train_features, valid_features,
             fold, model, optimizer, scheduler, device, config,
             exp_record,
             original_model, tokenizer,
             saving_ts, saving_dir, epoch,best_loss_sum,previous_best
            ):  
    
    best_metric = previous_best
    best_outputs_targets = (None, None)
    step = 0
    last_eval_step = 0
    last_train_eval_step = 0
    eval_period = config['EVAL_PERIOD']
    train_eval_period = config['TRAIN_EVAL_PERIOD']
    #training_log = config['TRAINING_LOG']
    
    losses = []
    train_total_meter = AccumulateMeter()
    train_steps_meter = AccumulateMeter()

    if config['FREEZE_EMBED']:
        for name, param in model.named_parameters():
            # embeddings.word_embeddings.weight
            #print(name)
            if 'embeddings.word_embeddings' in name:
                param.requires_grad = False
                
                
    GRAD_DESCD_STEP = True

    training_timer = Timer("training")
    train_steps_metrics_timer = Timer("train steps metrics")
    total_train_metrics_timer = Timer("train total metrics")
    eval_metrics_timer = Timer("eval metrics")
                    
    for idx, d in enumerate(data_loader):

        training_timer.start_timer()

        targets_start, targets_end = d['start_position'].to(device), d['end_position'].to(device)


        model_input_keys = ['input_ids', 'attention_mask']
        data = {key:val.reshape(val.shape[0],-1).to(device) for key,val in d.items() if key in model_input_keys}

        if GRAD_DESCD_STEP:
            optimizer.zero_grad()
        model.train()

        
        if config['AUTO_SCALER']:
            with torch.cuda.amp.autocast():
                outputs_start, outputs_end = model(**data)
        else:
            outputs_start, outputs_end = model(**data)


        loss = loss_fn((outputs_start, outputs_end), (targets_start, targets_end))
        loss = loss / config['GRAD_ACCU_STEPS']

        losses.append(loss.item())
        loss.backward()
        
        if config['GRAD_ACCU_STEPS'] != 1:
            optimizer.step()
            scheduler.step()
            GRAD_DESCD_STEP = True
        else:
            if idx % config['GRAD_ACCU_STEPS'] == 0 or idx == len(data_loader) - 1:
                optimizer.step()
                scheduler.step()
                GRAD_DESCD_STEP = True
            else:
                GRAD_DESCD_STEP = False
        
        last_lr = scheduler.get_last_lr()

        training_timer.stop_timer()

        # update meters
        train_steps_meter.update(d, outputs_start, outputs_end, targets_start, targets_end)
        train_total_meter.update(d, outputs_start, outputs_end, targets_start, targets_end)

        if step >= (last_train_eval_step + train_eval_period) or idx == (len(data_loader) -1):
            # Evaluate the model on train_loader.
            num_steps = step - last_eval_step
            last_train_eval_step = step

            with train_steps_metrics_timer:
                train_steps_metrics, train_steps_is_best, train_steps_last_best = train_steps_meter.get_metrics(tokenizer, config)
            with total_train_metrics_timer:
                if train_steps_meter.get_features_count() != train_total_meter.get_features_count():
                    train_total_metrics, train_total_is_best, train_total_last_best = train_total_meter.get_metrics(tokenizer, config)
                else:
                    train_total_metrics, train_total_is_best, train_total_last_best = train_steps_metrics, train_steps_is_best, train_steps_last_best

            logging.info(f'@desced step {step} @data step {idx} last lr: {min(last_lr)}-{max(last_lr)}\n'
                         f'Train Loss: {loss.item():.4f} Train Steps metrics(new best:{train_steps_is_best}) : {pprint_metrics(train_steps_metrics)}\n'
                         f'Train Total metrics(new best:{train_total_is_best}) : {pprint_metrics(train_total_metrics)}')
            train_steps_meter.reset()


        
        if step >= (last_eval_step + eval_period) or idx == (len(data_loader) -1):
            # Evaluate the model on val_loader.
            num_steps = step - last_eval_step
            last_eval_step = step

            with eval_metrics_timer:
                inter_eval_metrics, is_best, last_best = eval(valid_loader,model,device, config, best_metric, tokenizer)
            logging.info(f'@desced step {step} @data step {idx} last lr: {min(last_lr)}-{max(last_lr)}\n'
                         f'Train Loss: {loss.item():.4f} Val metrics(new best:{is_best}) : {pprint_metrics(inter_eval_metrics)}')
                
            if is_best:
                logging.info(f'!!new best Loss!!')
                new_best = inter_eval_metrics['jaccard']
                assert best_metric < new_best
                best_metric = new_best
                logging.info(f'{blu} Loss decreased from {last_best} -> {new_best}{blk}\n')

                # update record
                exp_record.update_fold(fold, new_best, loss.item())
                
                SAVING_LOSS_THRESHOLD = config['SAVING_THRESHOLD']
                
                if new_best > SAVING_LOSS_THRESHOLD:
                    save_run(original_model, model, tokenizer, config, saving_dir, fold)

                else:
                    logging.info(f"{new_best} weaker than saving threshold {SAVING_LOSS_THRESHOLD}, skip saving.")



        if GRAD_DESCD_STEP:
            step +=1

    logging.info(f"{training_timer.get_total_secs_str()}, {train_steps_metrics_timer.get_total_secs_str()}, "
                 f"{total_train_metrics_timer.get_total_secs_str()}, {eval_metrics_timer.get_total_secs_str()}")
    
    return exp_record, best_metric


# In[22]:


def eval(data_loader, model, device, config, previous_best, tokenizer):
    #logging.info(f"eval, data: {len(data_loader)}")
    model.eval()
    with torch.no_grad():
        meter = AccumulateMeter(previous_best = previous_best)

        for idx,d in enumerate(data_loader):

            targets_start, targets_end = d['start_position'].to(device), d['end_position'].to(device)

            model_input_keys = ['input_ids', 'attention_mask']
            data = {key: val.reshape(val.shape[0], -1).to(device) for key, val in d.items() if key in model_input_keys}


            
            if config['AUTO_SCALER']:
                with torch.cuda.amp.autocast():
                    outputs = model(**data)
            else:
                outputs = model(**data)
            
            outputs_start, outputs_end = outputs

            meter.update(d, outputs_start, outputs_end, targets_start, targets_end)

        #sp_cor = 0 #spearmanr(fin_targets, fin_outputs)
        
        # bootstrapping confident interval
        #rmse, low, high = bootstrap_rmse(fin_outputs, fin_targets, low_high=True)

        # get metrics for val:
        metrics, is_best, last_best = meter.get_metrics(tokenizer, config)
        
    return metrics, is_best, last_best

from sklearn import model_selection,metrics
import numpy as np
import transformers
import pprint
import shutil

def run(config, import_file_path=None):
    logging.info(f"running conf: {config}")
    #if not config['NEW_SEEDS']:
    #    seed_everything(config['SEED'])

    path = config['OUTPUT_ROOT_PATH']

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
        
    logging.info(f"using device:{device}")


    tokenizer = get_tokenizer(config)
    

    
    saving_ts = int(time.time())
    logging.info(f"saving_ts: {saving_ts}")
    saving_dir = f"{path}/pretrained-{saving_ts}"
    os.makedirs(saving_dir, exist_ok=True)
    logging.info(f"created saving dir: in {saving_dir}")

    # make sure no unpublished changes...
    repo = git.Repo(search_parent_directories=True)
    if not config['TEST_RUN']:
        assert len(repo.index.diff(None)) == 0, f"expect 0 unstage files, but there are: {len(repo.index.diff(None))}"
        uncommit_changes = len(repo.index.diff('HEAD'))
        assert uncommit_changes == 0, f"expect 0 unstage files, but there are: {uncommit_changes}"
        unpublished_changes = len(list(repo.iter_commits('main@{u}..main')))
        assert unpublished_changes == 0, f"expect no unpublished changes, but there are: {unpublished_changes}"


    #training_log = dict()
    #config.set('runtime', 'TRAINING_LOG', training_log)

    git_head_id = repo.head.object.hexsha
    #training_log['git_head_id'] = git_head_id
    logging.info(f"git head id {git_head_id}")
    
    best_loss_sum = 0.

    data, split_output = get_data_kfold_split(config)

        
    all_folds_outputs_targets = ([], [])

    exp_record = ExpRecord()
    exp_record.set_info(saving_ts, saving_dir, logging_file_path, git_head_id, config)

    for fold, (train_idx,valid_idx) in enumerate(tqdm(split_output, total=len(split_output))):

        train_loader, valid_loader, train_features, valid_features = make_loader(config, data, split_output, tokenizer, fold)
            
        if config['RESEED_EVERY_FOLD']:
            seed_everything(config['SEED'] + fold)
        if config['STOP_AT_FOLD'] == fold:
            logging.info(f"stopping at {fold}...")
            break
        start_time = time.time()
        
        logging.info(f"training config: {pprint.pformat(config)}")
        logging.info(f"========== USING {device} ==========")
        logging.info(f'========== Fold: {fold} ==========')

        
        model_class = getattr(model_import, config['MODEL_CLASS'])


        if config['PRETRAIN_TO_LOAD'] is not None:
            original_model = model_class(from_pretrain=config['PRETRAIN_TO_LOAD'],
                                         config=config)
        else:
            original_model = model_class(config=config)
            

            
        if config['HEAD_DROPOUT'] is not None:
            assert "(head_dropout): Dropout" in str(original_model)

            
        if not config['EMBED_OTHER_GPU']:
            original_model.to(device)
        
        if config['GPU_PARALLEL_IDS'] is not None:
            logging.info(f"using device ids: {config['GPU_PARALLEL_IDS']}")
            model = torch.nn.DataParallel(original_model, device_ids=config['GPU_PARALLEL_IDS'])
        else:
            
            model = original_model
            

        optimizer = make_optimizer(original_model, config)
        scheduler = make_scheduler(optimizer, train_loader, config)
            
        best_metric = 0
        


            
            
        torch.cuda.empty_cache()
                
        for epoch in range(config['EPOCHS']):


            logging.info(f'========== epoch : {epoch+1}==========')
            rec, best_metric = train_fn(train_loader, valid_loader,
                                        train_features, valid_features,
                                        fold,
                                        model,
                                        optimizer,
                                        scheduler,
                                        device,
                                        config,
                                        exp_record,
                                        original_model, tokenizer,
                                        saving_ts, saving_dir, epoch, best_loss_sum,
                                        previous_best=best_metric)
        
        end_time = time.time()
        elp_fold = end_time - start_time
        logging.info(f'===== Fold Time: {elp_fold} =====')

        logging.info(f"fold record: {exp_record}")

        # cleanup after fold is done
        logging.info(f'cleanup after fold is done')
        del model
        del original_model
        gc.collect()
        torch.cuda.empty_cache()
        
    logging.info(f"run done with jaccard: {exp_record.get_mean_jaccard()}")
    exp_record.persist()
    return saving_ts, saving_dir, exp_record.get_mean_jaccard(), exp_record


