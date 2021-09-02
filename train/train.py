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

run_ts = int(time.time())

TRAIN_MODE = True
TEST_MODE = False


# In[5]:


# set up logging:


import sys
sys.path.append('../')

from utils.utils import *
from loss import *
from data import *
from optimizer import *
from model.model import *

logging = utils.logging


# In[2]:


import numpy as np
from scipy.stats import norm
from scipy import stats
# referrencing: https://knowledge-repo.d.musta.ch/post/projects/datau332_recalculating_erf_metrics.kp



#import_file_name = 'model/model.py'

#model_import = __import__(import_file_name)




                
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



def get_train_test():

    input_path = "../input/"

    train = pd.read_csv(f'{input_path}chaii-hindi-and-tamil-question-answering/train.csv')
    test = pd.read_csv(f'{input_path}chaii-hindi-and-tamil-question-answering/test.csv')
    external_mlqa = pd.read_csv(f'{input_path}mlqa-hindi-processed/mlqa_hindi.csv')
    external_xquad = pd.read_csv(f'{input_path}mlqa-hindi-processed/xquad.csv')
    external_train = pd.concat([external_mlqa, external_xquad])






    path = 'crp/data/' if in_private_env else '../input/commonlitreadabilityprize/'
    train = pd.read_csv(path + 'train.csv')
    train = train[(train.target != 0) & (train.standard_error != 0)].reset_index(drop=True)
    test = pd.read_csv(path + 'test.csv')
    return train, test



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



def train_fn(data_loader, valid_loader, train_x, fold, model, optimizer, scheduler, device, config, 
             original_model, tokenizer,
             saving_ts, saving_dir, epoch,best_loss_sum,previous_best_loss
            ):  
    
    EVAL_SCHEDULE = [(0.50, 16), (0.49, 8), (0.48, 4), (0.47, 2), (-1., 1)]
    best_val_rmse = previous_best_loss
    best_epoch = 0
    best_outputs_targets = (None, None)
    step = 0
    last_eval_step = 0
    eval_period = EVAL_SCHEDULE[0][1]    
    training_log = config.get('runtime', 'TRAINING_LOG')
    
    losses = []
    
    dumped_data = 0
    
    if config.getboolean(configparser.DEFAULTSECT, 'FREEZE_EMBED', fallback=False):
        for name, param in model.named_parameters():
            # embeddings.word_embeddings.weight
            #print(name)
            if 'embeddings.word_embeddings' in name:
                param.requires_grad = False
                
                
    GRAD_DESCD_STEP = True
                    
    for idx, d in enumerate(data_loader):

        if len(d) == 3:
            if DEBUG_PRINT:
                data,targets, excerpt = d
                weights = None
            else:
                data,targets,weights = d
        else:
            data,targets = d
            weights = None
            
        data = {key:val.reshape(val.shape[0],-1).to(device) for key,val in data.items()}
        
        if config.getboolean(configparser.DEFAULTSECT,'REMOVE_TOKEN_TYPES', fallback=False) and 'token_type_ids' in data:
            del data['token_type_ids']
        
        targets = targets.to(device)
        if weights is not None:
            weights = weights.to(device)

        if GRAD_DESCD_STEP:
            optimizer.zero_grad()
        model.train()
        
        if DEBUG_PRINT and dumped_data < 5:
            print(f"data inputed to model: {(data['input_ids'], data['attention_mask'])}")
            print(f"data inputed to model len: {len(data['attention_mask'])}")
            print(excerpt)
            dumped_data += 1
        
        if config.getboolean(configparser.DEFAULTSECT, 'AUTO_SCALER', fallback=False):
            with torch.cuda.amp.autocast():
                outputs, _ = model(**data)
        else:
            outputs, _ = model(**data)
        outputs = outputs.squeeze(-1)
        #Eprint(outputs)
        
        loss = loss_fn(outputs, targets, config, weights)
        loss = loss / config.getint(configparser.DEFAULTSECT, 'GRAD_ACCU_STEPS', fallback=1)

        losses.append(loss.item())
        loss.backward()
        
        if config.getint(configparser.DEFAULTSECT, 'GRAD_ACCU_STEPS', fallback=1) != 1:
            optimizer.step()
            scheduler.step()
            GRAD_DESCD_STEP = True
        else:
            if idx % config.getint(configparser.DEFAULTSECT, 'GRAD_ACCU_STEPS', fallback=1) == 0 or idx == len(data_loader) - 1:
                optimizer.step()
                scheduler.step()
                GRAD_DESCD_STEP = True
            else:
                GRAD_DESCD_STEP = False
        
        last_lr = scheduler.get_last_lr()
        
        
        #nm = 1
        #if step == nm:
        #    torch.save(model.state_dict(), "/tmp/b"+str(nm))
        #    torch.save(data['input_ids'], "/tmp/binput_ids"+str(nm))
        #    torch.save(data['attention_mask'], "/tmp/battention_mask"+str(nm))
        #    torch.save(targets, "/tmp/btarget"+str(nm))
        #    print(f"model hash saved in /tmp/b"+str(nm))
        
        if step >= (last_eval_step + eval_period):
            # Evaluate the model on val_loader.
            num_steps = step - last_eval_step
            last_eval_step = step
                
            #val_rmse = math.sqrt(eval_mse(model, val_loader))  
            val_rmse, outputs, eval_targets, spr_cor, low, high = eval(valid_loader,model,device, config, train_x)
            logging.info(f'@desced step {step} @data step {idx} last lr: {min(last_lr)}-{max(last_lr)} Train Loss: {loss} Val Loss : {val_rmse} - ({low},{high}), Spearman Corr: {spr_cor}')
            #losses_valid.append(loss)


            for rmse, period in EVAL_SCHEDULE:
                if val_rmse >= rmse:
                    eval_period = period
                    break                               
                
            if not best_val_rmse or val_rmse < best_val_rmse:  
                logging.info(f'!!new best Loss!!')
                logging.info(f'{blu} Loss decreased from {best_val_rmse} -> {val_rmse}{blk}\n')
                training_log['saving_ts'] = saving_ts
                training_log['saving_dir'] = saving_dir
                training_log[f'fold_{fold}'] = dict()
                training_log[f'fold_{fold}']['best_epoch'] = epoch
                training_log[f'fold_{fold}']['best_loss'] = str(val_rmse)
                training_log[f'fold_{fold}']['total_loss'] = str((best_loss_sum+val_rmse)/(fold+1))
                training_log[f'fold_{fold}']['spearman_corr'] = spr_cor
                training_log['logging_file'] = logging_file_path
                
                SAVING_LOSS_THRESHOLD = 0.6
                
                if val_rmse < SAVING_LOSS_THRESHOLD:

                    original_model.save_pretrained(model, f'{saving_dir}/model_{fold}')
                    model_import.save_training_config(config, f'{saving_dir}/model_{fold}')
                    tokenizer.save_pretrained(f'{saving_dir}/model_{fold}/tokenizer')

                    # save config another copy in tokenizer
                    original_model.roberta.config.save_pretrained(f'{saving_dir}/model_{fold}/tokenizer')
                    logging.info(f"saved in {saving_dir}/model_{fold}")
                    print(f"saved in {saving_dir}/model_{fold}")
                else:
                    logging.info(f"{val_rmse} smaller than saving threshold {SAVING_LOSS_THRESHOLD}, skip saving.")
                    
                best_preds = outputs
        #             torch.save(model.state_dict(), config['MODEL_PATH'])
                best_val_rmse = val_rmse  
                best_outputs_targets = (outputs, eval_targets)
                
                
        if GRAD_DESCD_STEP:
            step +=1
    
    return best_val_rmse, best_outputs_targets


# In[22]:


def eval(data_loader, model, device, config, train_df=None):
    model.eval()
    with torch.no_grad():
        fin_targets = []
        fin_outputs = []
        fin_ids = []
        for idx,to_upack in enumerate(data_loader):
            data,targets = to_upack
            data = {key:val.reshape(val.shape[0],-1).to(device) for key,val in data.items()}
            
            if config.getboolean(configparser.DEFAULTSECT,'REMOVE_TOKEN_TYPES', fallback=False) and 'token_type_ids' in data:
                del data['token_type_ids']
            
            targets = targets.to(device)
            
            if config.getboolean(configparser.DEFAULTSECT, 'AUTO_SCALER', fallback=False):
                with torch.cuda.amp.autocast():
                    outputs, _ = model(**data)
            else:
                outputs, _ = model(**data)
            
            outputs = outputs.squeeze(-1)
            
            #outputs = outputs["logits"].squeeze(-1)

            if config.get(configparser.DEFAULTSECT,'LOSS_TYPE', fallback=None) == 'multi-class':
                outputs = bin_values[torch.argmax(outputs, dim=-1).cpu().detach().numpy()]
                fin_outputs.extend(outputs.tolist())
                
            
#             targets = data['targets']

#             outputs = model(data['input_ids'], data['attention_mask'])
            else:
        
                fin_outputs.extend(outputs.detach().cpu().numpy().tolist())
            
            fin_targets.extend(targets.detach().cpu().detach().numpy().tolist())

        #loss = loss_fn(torch.tensor(fin_outputs),torch.tensor(fin_targets), config, loss_type='sqrt_mse')
        
        # calculate spearman corr:
        #fin_targets.extend(train_df.target.tolist())
        #fin_outputs.extend(train_df.target.tolist())
        
        sp_cor = 0 #spearmanr(fin_targets, fin_outputs)
        
        # bootstrapping confident interval
        rmse, low, high = bootstrap_rmse(fin_outputs, fin_targets, low_high=True)
        
    return rmse,fin_outputs, fin_targets, sp_cor, low, high 


# for debug...
printed_debug_info = True

def infer(data_loader, model, device, config, return_embed = False, use_tqdm=True):
    global printed_debug_info
    printed_debug_info = True
    model.eval()
    with torch.no_grad():
        fin_targets = []
        fin_outputs = []
        if return_embed:
            fin_outputs_embed = []
        to_for = enumerate(data_loader)
        if use_tqdm:
            to_for = tqdm(to_for, total = len(data_loader))
        for idx, data in to_for:
            data = {key:val.reshape(val.shape[0],-1).to(device) for key,val in data.items()}
            
            
            if not printed_debug_info:
                print(f"input to model:{data}")
                print(f"input to model shape:{data['attention_mask'].shape}")
                
            outputs, embeds = model(**data)
            if not printed_debug_info:
                print(f"output to model:{outputs}")
                print(f"output to model shape:{outputs.shape}")
                
                printed_debug_info = True
            outputs = outputs.squeeze(-1)
            
            #outputs = outputs["logits"].squeeze(-1)
            
            if config.get(configparser.DEFAULTSECT,'LOSS_TYPE', fallback=None) == 'multi-class':
                outputs = torch.argmax(outputs, dim=-1)

#             targets = data['targets']

#             outputs = model(data['input_ids'], data['attention_mask'])
            
            fin_outputs.extend(outputs.detach().cpu().numpy().tolist())
            if return_embed:
                fin_outputs_embed.extend(embeds.detach().cpu().numpy())
    if return_embed:
        return fin_outputs, fin_outputs_embed 
    else:
        return fin_outputs


# In[ ]:

# In[23]:


from sklearn import model_selection,metrics
import numpy as np
import transformers
from transformers import AdamW
import pprint
from transformers import get_linear_schedule_with_warmup
import shutil

def run(config, import_file_path=None):
    print(f"running conf: {config}")
    #if not config['NEW_SEEDS']:
    #    seed_everything(config['SEED'])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
        
    print(f"using device:{device}")
    logging.info(f"using device:{device}")


    tokenizer = get_tokenizer(config)
    
    if config.getboolean(configparser.DEFAULTSECT, 'STRATEFIED', fallback=False):
        kfold = StratifiedKFold(n_splits=config.getint(configparser.DEFAULTSECT, 'FOLDS', fallback=3),
                                shuffle=True,
                                random_state=config.getint(configparser.DEFAULTSECT, 'SEED', fallback=42))
    else:
        kfold = KFold(n_splits=config.getint(configparser.DEFAULTSECT, 'FOLDS', fallback=3),
                      random_state=config.getint(configparser.DEFAULTSECT, 'SEED', fallback=42),
                      shuffle=True)
    
    saving_ts = int(time.time())
    logging.info(f"saving_ts: {saving_ts}")
    saving_dir = f"{path}/pretrained-{saving_ts}"
    os.makedirs(saving_dir, exist_ok=True)
    logging.info(f"created saving dir: in {saving_dir}")
    
    if import_file_path is not None:
        if isinstance(import_file_path, list):
            for l in import_file_path:
                shutil.copy2(l,f"{saving_dir}/")
        else:
            shutil.copy2(import_file_path,f"{saving_dir}/")
    
    training_log = dict()
    config.set('runtime', 'TRAINING_LOG', training_log)
    
    best_loss_sum = 0.
    
    if config.getboolean(configparser.DEFAULTSECT, 'STRATEFIED', fallback=False):
        split_output = kfold.split(X=train,y=bins)
    else:
        split_output = kfold.split(train)
        
    all_folds_outputs_targets = ([], [])
    
    for fold , (train_idx,valid_idx) in enumerate(tqdm(split_output, total=config['FOLDS'])):
            
        if config.getboolean(configparser.DEFAULTSECT, 'NEW_SEEDS', fallback=False):
            seed_everything(config.getint(configparser.DEFAULTSECT, 'SEED', fallback=42) + fold)
        if config.getint(configparser.DEFAULTSECT, 'STOP_AT_FOLD', fallback=-1) == fold:
            logging.info(f"stopping at {fold}...")
            break
        start_time = time.time()
        train_x,valid_x = train.loc[train_idx],train.loc[valid_idx]

        add_augment_conf = config.get(configparser.DEFAULTSECT, 'ADD_AUGMENT', fallback=None)
        
        if add_augment_conf is not None and config.getboolean(configparser.DEFAULTSECT, 'AUGMENT_SKIP_TRAINING', fallback=False):
            train_x = train_x.drop(train_x.index.values)
        
        if add_augment_conf is not None and isinstance(add_augment_conf, str):
            train_aug = pd.read_csv(add_augment_conf)
            # exclude ids in val:
            print(f"before exclude ids in val len train_aug: {len(train_aug)}")
            train_aug = train_aug[~train_aug.id.isin(valid_x.id.values)]
            print(f"after exclude ids in val len train_aug: {len(train_aug)}")
            train_x = train_x.append(train_aug).sample(frac=1).reset_index(drop=True)
        elif add_augment_conf is not None and isinstance(add_augment_conf, list):
            for aug in add_augment_conf:
                train_aug = pd.read_csv(aug)
                # exclude ids in val:
                print(f"before exclude ids in val len train_aug: {len(train_aug)}")
                train_aug = train_aug[~train_aug.id.isin(valid_x.id.values)]
                print(f"after exclude ids in val len train_aug: {len(train_aug)}")
                train_x = train_x.append(train_aug).sample(frac=1).reset_index(drop=True)

        else:
            train_x = train_x.reset_index(drop=True)
            
        if config.getboolean(configparser.DEFAULTSECT, 'AUGMENT_REWEIGHTING', fallback=False) and \
            add_augment_conf is not None:
            id_reweighting_df = train_x.groupby('id', as_index=False).agg(reweighting=pd.NamedAgg(column="excerpt", aggfunc="count"))
            train_x = train_x.merge(id_reweighting_df, on='id', how='left')
            train_x['reweighting'] = 1. / train_x['reweighting']
            
            assert train_x.groupby('id').agg(reweighting_sum=pd.NamedAgg(column='reweighting',aggfunc='sum'))['reweighting_sum'].apply(lambda x: np.isclose(x, 1.0)).all()
            
        valid_x = valid_x.reset_index(drop=True)

        train_ds = CommonLitDataset(train_x, tokenizer, config, add_augment_conf)
        
        multi_gpu_batch_size = 1
        
        if config.get('runtime', 'GPU_PARALLEL_IDS', fallback=None) is not None:
            multi_gpu_batch_size = len(config.get('runtime', 'GPU_PARALLEL_IDS', fallback=None))

        if not config.getboolean(configparser.DEFAULTSECT, 'STRATEFIED_SAMPLER', fallback=False):
            train_loader = torch.utils.data.DataLoader(
                train_ds,
                batch_size = config.getint(configparser.DEFAULTSECT, 'TRAIN_BATCH_SIZE', fallback=16) * multi_gpu_batch_size,
                num_workers = 2,
                shuffle = config.getboolean(configparser.DEFAULTSECT, 'SHUFFLE_TRAIN', fallback=True),
                drop_last=True,
            )
        else:
            #y, batch_size, shuffle=True, random_state=42
            sampler = StratifiedBatchSampler(train_x['bins'], 
                                             batch_size=config.getint(configparser.DEFAULTSECT, 'TRAIN_BATCH_SIZE', fallback=16) * multi_gpu_batch_size,
                                             shuffle=config.getboolean(configparser.DEFAULTSECT, 'SHUFFLE_TRAIN', fallback=True),
                                             random_state=config.getint(configparser.DEFAULTSECT, 'SEED', fallback=42)
                                            )
            train_loader = torch.utils.data.DataLoader(
                train_ds,
                batch_sampler = sampler,
                num_workers = 2,
            )

        valid_ds = CommonLitDataset(valid_x, tokenizer, config)

        valid_loader = torch.utils.data.DataLoader(
            valid_ds,
            batch_size = config.getint(configparser.DEFAULTSECT, 'VALID_BATCH_SIZE', fallback=16)  * multi_gpu_batch_size,
            num_workers = 2,
            drop_last=False,
        )
        
        if config.getboolean(configparser.DEFAULTSECT, 'NEW_SEEDS', fallback=False):
            seed_everything(config.getint(configparser.DEFAULTSECT, 'SEED', fallback=42) + fold)
        
        #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        logging.info(f"training config: {pprint.pformat(config)}")
        logging.info(f"========== USING {device} ==========")
        logging.info(f'========== Fold: {fold} ==========')
        num_labels=1
        if config.get(configparser.DEFAULTSECT,'LOSS_TYPE', fallback=None) == 'multi-class':
            num_labels = config.getint(configparser.DEFAULTSECT, 'BINS_COUNT', fallback=16)
        #original_model = AutoModelForSequenceClassification.from_pretrained(config['BERT_PATH'],num_labels=num_labels)
        
        model_class = getattr(model_import, config.get(configparser.DEFAULTSECT, 'MODEL_CLASS', fallback=None))



        if config.get(configparser.DEFAULTSECT,'PRETRAIN_TO_LOAD', fallback=None) is not None:
            original_model = model_class(from_pretrain=config.get(configparser.DEFAULTSECT,'PRETRAIN_TO_LOAD', fallback=None),
                                         model_config=config)
        else:
            original_model = model_class(model_config=config)
            

            
        if config.getboolean(configparser.DEFAULTSECT, 'FIX_DROPOUT', fallback=True) and \
                config.getfloat(configparser.DEFAULTSECT, 'HEAD_DROPOUT', fallback=0) is not None:
            assert "(head_dropout): Dropout" in str(original_model)
            #print(f"dropout on, dump model: {original_model}")

        #torch.save(original_model.state_dict(), "/tmp/b0")
        #print(f"model hash saved in /tmp/b0")
            
        if config.getfloat(configparser.DEFAULTSECT, 'EMBED_OTHER_GPU', fallback=None) is None:
            original_model.to(device)
        
        if config.get('runtime', 'GPU_PARALLEL_IDS', fallback=None) is not None:
            print(f"using device ids: {config.get('runtime', 'GPU_PARALLEL_IDS', fallback=None)}")
            logging.info(f"using device ids: {config.get('runtime', 'GPU_PARALLEL_IDS', fallback=None)}")
            model =  torch.nn.DataParallel(original_model, device_ids=config.get('runtime', 'GPU_PARALLEL_IDS', fallback=None))
        else:
            
            model = original_model
            
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias','LayerNorm.bias','LayerNorm.weight']
        optimizer_parameters = [
            {'params' : [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay' : 0.001},
            {'params' : [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay' : 0.0},
        ]

        num_train_steps = int(len(train_ds) / config.getint(configparser.DEFAULTSECT, 'TRAIN_BATCH_SIZE', fallback=16)
                              * config.getint(configparser.DEFAULTSECT, 'EPOCHS', fallback=3))
        
        if config.getboolean(configparser.DEFAULTSECT, 'FIX_STEPS_BUG', fallback=True):
            grad_accu_factor = 1

            grad_accu_factor = config.getint(configparser.DEFAULTSECT, 'GRAD_ACCU_STEPS', fallback=1)
            num_train_steps = int(len(train_loader) * config.getint(configparser.DEFAULTSECT, 'EPOCHS', fallback=3) *
                                  config.getint(configparser.DEFAULTSECT, 'STEPS_FACTOR', fallback=1) / grad_accu_factor)

            

#         optimizer = AdamW(optimizer_parameters, lr = 3e-5, betas=(0.9, 0.999))
        if config.getboolean(configparser.DEFAULTSECT, 'USE_SIMPLE_OPTIMIZER', fallback=False):
            optimizer = AdamW(model.parameters(), lr = config.getfloat(configparser.DEFAULTSECT, 'LR', fallback=1e-5), betas=(0.9, 0.999),
                              weight_decay=config.getfloat(configparser.DEFAULTSECT, 'ADAM_WEIGHT_DECAY', fallback=1e-5)#1e-5
                             )
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps = int(config.getfloat(configparser.DEFAULTSECT, 'WARMUP_STEPS_RATIO', fallback=0) * num_train_steps),
                num_training_steps = num_train_steps
            )
#         scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=len(train_ds), max_lr=1e-4, epochs=config['EPOCHS'])
        else:
            optimizer = make_optimizer(original_model, config)
            scheduler = make_scheduler(optimizer, train_ds, config, train_loader)
            
        best_loss = 99999
        
        losses_valid = list()
        best_preds = list()

        if config.getint(configparser.DEFAULTSECT, 'VAL_STEPS_CHUNKS', fallback=None) is not None:
            num_steps = total_steps // config.getint(configparser.DEFAULTSECT, 'VAL_STEPS_CHUNKS', fallback=None)
        
        if config.getint(configparser.DEFAULTSECT, 'SCHEDULED_EVAL', fallback=None) is not None and \
                config.getint(configparser.DEFAULTSECT, 'SCHEDULED_EVAL', fallback=None):
            EVAL_SCHEDULE = [(0.50, 16), (0.49, 8), (0.48, 4), (0.47, 2), (-1., 1)]
            num_steps = EVAL_SCHEDULE[0][1]
            
            
        torch.cuda.empty_cache()
                
        for epoch in range(config.getint(configparser.DEFAULTSECT, 'EPOCHS', fallback=3)):
            current_step = 0
            
            chunk_id = 0
            
                    
            start = time.time()

            logging.info(f'========== epoch : {epoch+1}==========')
            best_loss, best_outputs_targets = train_fn(train_loader, valid_loader, train_x,fold,
                                          model, 
                                          optimizer,
                                          scheduler,
                                          device, 
                                          config, 
                                          original_model, tokenizer,
                                          saving_ts, saving_dir, epoch, best_loss_sum, previous_best_loss=best_loss)
                

        if best_outputs_targets[0] is None or best_outputs_targets[1] is None:
            print(f"best_outputs_targets None: {best_outputs_targets}")
        else:
            
            all_folds_outputs_targets[0].extend(best_outputs_targets[0])
            all_folds_outputs_targets[1].extend(best_outputs_targets[1])
        
        all_folds_loss = bootstrap_rmse(all_folds_outputs_targets[0], all_folds_outputs_targets[1], low_high=False)[0]
        
        print(f"all folds len: {len(all_folds_outputs_targets[0])}/ all len: {len(train)}")
        
        end_time = time.time()
        elp_fold = end_time - start_time
        logging.info(f'===== Fold Time: {elp_fold} =====')
        
        best_loss_sum += best_loss
        
        logging.info(f"\n saving_ts:{saving_ts}, total loss: {best_loss_sum/(fold+1)}, all folds: {all_folds_loss}")
        print(f"\n saving_ts:{saving_ts}, total loss: {best_loss_sum/(fold+1)}, all folds: {all_folds_loss}")
        logging.info(training_log)
        print(training_log)
        
        # cleanup after fold is done
        logging.info(f'cleanup after fold is done')
        del model
        del original_model
        gc.collect()
        torch.cuda.empty_cache()
        
    print(f"run done with loss: {best_loss_sum/(fold+1)}")
    logging.info(f"run done with loss: {best_loss_sum/(fold+1)}")
    return saving_ts, saving_dir, best_loss_sum/(fold+1), all_folds_loss




    
def pred_df(df, config, upload_name='pretrained-model-1621892031'):
    pretrain_base_path = f'crp/data/{upload_name}/' if 'gavin_li' in cwd else f'../input/{upload_name}/'
    
    if 'SEED' in config:
        seed_everything(config.getint(configparser.DEFAULTSECT, 'SEED', fallback=42))
    else:
        seed_everything(43)
    
    #print(pretrain_base_path)

    import json
    
    # assert the save version...
    
    #with open(f"{pretrain_base_path}/model_0/training_config.json", 'r') as f:
    #    conf = json.load(f)
    #    if 'saving_ts' in conf:
    #        assert conf['saving_ts'] == 1622060846, 'saving_ts should match with the saved one'
        
        
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    import os

    from pathlib import Path
    from transformers.file_utils import WEIGHTS_NAME
    pretrain_paths = []

    pathlist = Path(pretrain_base_path).glob('**/*')
    for path in pathlist:
        # because path is object not string
        path_in_str = str(path)
        # print(path_in_str)
        if 'model' in path_in_str and os.path.isdir(path_in_str) and path_in_str[-7:-1] == 'model_':
            pretrain_paths.append(path_in_str)

    pretrain_paths.sort()
    print(pretrain_paths)
    pred_sum = np.zeros((len(df)))

    print(f'loadding tokenizer from {pretrain_paths[0]}')

    # fix GPT2 save pretrain issue...
    print(f"loading tokenizer for {config.get(configparser.DEFAULTSECT, 'TOKENIZER', fallback=None)}")
    if config.get(configparser.DEFAULTSECT, 'TOKENIZER', fallback=None) == 'microsoft/deberta-base':
        print(f"fix GPT2 save pretrain issue by direct load from huggingface registry...")
        if os.path.exists("../input/deberta-base-uncased-tokenizer"):
            print(f"loading from ../input/deberta-base-uncased-tokenizer")
            tokenizer = DebertaTokenizer.from_pretrained(f"../input/deberta-base-uncased-tokenizer")
        elif os.path.exists(f"{str(pretrain_paths[0])}/tokenizer/"):
            print(f"loading from {pretrain_paths[0]}/tokenizer/")
            tokenizer = DebertaTokenizer.from_pretrained(f"{str(pretrain_paths[0])}/tokenizer/")
        else:
            print(f"loading from {pretrain_paths[0]}")
            tokenizer = DebertaTokenizer.from_pretrained(f"{str(pretrain_paths[0])}")
    else:
        if os.path.exists(f"{str(pretrain_paths[0])}/tokenizer/"):
            if config.get(configparser.DEFAULTSECT, 'TOKENIZER', fallback=None) in ["deepset/roberta-base-squad2","chkla/roberta-argument","roberta-base", "deepset/roberta-large-squad2"]:
                tmp_ts = int(time.time())
                os.system(f'mkdir tmp_{tmp_ts}')
                tokenizer_dir = f"{str(pretrain_paths[0])}/tokenizer/"
                from_dir = f"{str(pretrain_paths[0])}/"
                os.system(f'cp {tokenizer_dir}/* tmp_{tmp_ts}/')
                os.system(f'cp {from_dir}/config.json tmp_{tmp_ts}/')
                tokenizer = AutoTokenizer.from_pretrained(f"tmp_{tmp_ts}/")
            else:
                print(f"loading from {pretrain_paths[0]}/tokenizer/")
                tokenizer = AutoTokenizer.from_pretrained(f"{str(pretrain_paths[0])}/tokenizer/")
        else:
            print(f"loading from {pretrain_paths[0]}")
            tokenizer = AutoTokenizer.from_pretrained(f"{str(pretrain_paths[0])}")

    sub_ds = CommonLitDataset(df, tokenizer, config)    
    
    sub_ds_loader = torch.utils.data.DataLoader(
            sub_ds,
            batch_size = config.getint(configparser.DEFAULTSECT, 'VALID_BATCH_SIZE', fallback=16),
            num_workers = 2,
            drop_last=False,
        )
    
    
    scores = []
    embeds = []

    for p in pretrain_paths:

        #model = AutoModelForSequenceClassification.from_pretrained(p,num_labels=1)
        model_class = getattr(model_import, config.get(configparser.DEFAULTSECT, 'MODEL_CLASS', fallback=None))
        
        # hardcode upload fix:
        if "single-model-v7-1627797474/model_3" in p:
            model_config = json.load(open(f'../input/single-model-v7-1627797474/model_2/training_config.json'))
            subprocess.getoutput("mkdir /tmp/model_3")
            subprocess.getoutput("cp ../input/single-model-v7-1627797474/model_3/*.bin /tmp/model_3/")
            subprocess.getoutput("cp ../input/single-model-v7-1627797474/model_2/*.json /tmp/model_3/")
            subprocess.getoutput("cp -r ../input/single-model-v7-1627797474/model_2/tokenizer /tmp/model_3/")
        else:
            
            model_config = json.load(open(f'{p}/training_config.json'))
            
        model_config = {**base_config, **model_config}
        
        print(f"loading model class:{config['MODEL_CLASS']}\n pretrain: {str(p)}, config:{model_config}")
      
        if model_config['EMBED_OTHER_GPU'] is not None:
            model_config['EMBED_OTHER_GPU'] = 0

            
        if "single-model-v7-1627797474/model_3" in p:
            model = model_class(from_pretrain='/tmp/model_3', model_config=model_config)
            model.load_checkpoint('/tmp/model_3')
        else:
            model = model_class(from_pretrain=p, model_config=model_config)
            model.load_checkpoint(p)
        model.to(device)
        
        #print('loading checkpoint...')
    
        #checkpoint = torch.load(os.path.join(p, WEIGHTS_NAME), map_location=torch.device('cpu'))
        #model.load_state_dict(checkpoint,strict=False)

        #print(f'{os.path.join(p, WEIGHTS_NAME)} loaded.')

        
        #print(f"input to infer:{df}")
        outputs, output_embeds = infer(sub_ds_loader,model,device, model_config, return_embed=True)
        #print(f"output to infer:{output_embeds}")

        #print(sum)
        print(outputs)

        pred_sum += outputs
        scores.append(outputs)
        embeds.append(output_embeds)
        
        
        # cleanup after fold is done
        print(f'cleanup after model is done')
        del model
        gc.collect()
        torch.cuda.empty_cache()
        

    pred_sum = pred_sum/(len(pretrain_paths))
    return pred_sum, scores, embeds


# In[ ]:

from sklearn.metrics import mean_squared_error

def create_submission(_,predictions, calibrate_rms=None):
    df =_.copy()
    df['target'] = predictions
    if calibrate_rms is not None:
        x = mean_squared_error(predictions, np.zeros(len(df['target'])), squared=False)
        df['target'] = predictions / x * calibrate_rms
        
    return df[['id','target']]


# In[ ]:


def gen_submission(TRAIN_MODE=False, TEST_ON_TRAINING=True, gen_file=True):
    to_ret = None
    if not TRAIN_MODE:
        if TEST_ON_TRAINING:
            # test first...
            pred_sum_train, _, _ = pred_df(train[['excerpt','id']], config)


            loss_on_train = loss_fn(torch.tensor(pred_sum_train), torch.tensor(train['target'].values), config, loss_type='sqrt_mse').item()
            assert loss_on_train < 0.55

            print(f"loss on training: {loss_on_train}")

        pred_sum, _, _ = pred_df(test, config)
        
        to_ret = pred_sum


        pred = create_submission(test, pred_sum)
        print(pred.head())
    else:
        # test infer on training set when it's training mode...
        pred_sum_train, _, _ = pred_df(train[['excerpt','id']], config)
        loss_on_train = loss_fn(torch.tensor(pred_sum_train), torch.tensor(train['target'].values), config, loss_type='sqrt_mse').item()
        assert loss_on_train < 0.55, f"{loss_on_train} shoudl be small"

        print(f"loss on training: {loss_on_train}")
        to_ret = pred_sum_train

        
    if not TRAIN_MODE and gen_file:
        pred.to_csv('./submission.csv',index=False)
        
    return to_ret


# In[ ]:

def average_ensemble(ts_to_scores):
    keys = list(ts_to_scores.keys())
    pred_sum = np.zeros(len(ts_to_scores[keys[0]]))
    for k, v in ts_to_scores.items():
        pred_sum += v

    pred_sum = pred_sum/len(keys)
    return pred_sum

def fit_ensemble_preds_then_ensemble(fit_ensemble_preds, ts_to_scores):
    model_preds = pd.read_csv(fit_ensemble_preds)
    
    from sklearn.linear_model import RidgeCV
    import math
    #X, y = load_diabetes(return_X_y=True)
    alphas=[1e-3, 1e-2, 1e-1, 1]

    clf = RidgeCV().fit(model_preds[[col for col in model_preds.columns if col[:5] == "pred_"]], model_preds['target'])
    #clf.score(X, y)
    print(f"fit best score: {math.sqrt(-clf.best_score_)}")
    
    
    keys = list(ts_to_scores.keys())
    for k in keys:
        assert f"pred_{k}" in model_preds.columns
    for col in model_preds.columns:
        if col[:5] == "pred_":
            assert int(col[5:]) in keys
            
    return clf.predict(np.hstack([ts_to_scores[int(col[5:])].reshape(-1,1) for col in model_preds.columns if col[:5] == "pred_"]))
    #return clf.predict([ts_to_scores[int(col[5:])] for col in model_preds.columns if col[:5] == "pred_"])

import pickle

def get_upload_dir(ts):
    if in_private_env:
        return f"pretrained-{ts}"
    
    if os.path.exists(f"../input/single-model-v14-{ts}"):
        return f"single-model-v14-{ts}"
    elif os.path.exists(f"../input/single-model-v13-{ts}"):
        return f"single-model-v13-{ts}"
    elif os.path.exists(f"../input/single-model-v12-{ts}"):
        return f"single-model-v12-{ts}"
    elif os.path.exists(f"../input/single-model-v11-{ts}"):
        return f"single-model-v11-{ts}"
    elif os.path.exists(f"../input/single-model-v10-{ts}"):
        return f"single-model-v10-{ts}"
    elif os.path.exists(f"../input/single-model-v9-{ts}"):
        return f"single-model-v9-{ts}"
    elif os.path.exists(f"../input/single-model-v8-{ts}"):
        return f"single-model-v8-{ts}"
    elif os.path.exists(f"../input/single-model-v7-{ts}"):
        return f"single-model-v7-{ts}"
    elif os.path.exists(f"../input/single-model-v6-{ts}"):
        return f"single-model-v6-{ts}"
    elif os.path.exists(f"../input/single-model-v5-{ts}"):
        return f"single-model-v5-{ts}"
    elif os.path.exists(f"../input/single-model-v4-{ts}"):
        return f"single-model-v4-{ts}"
    elif os.path.exists(f"../input/single-model-v3-{ts}"):
        return f"single-model-v3-{ts}"
    elif os.path.exists(f"../input/single-model-v2-{ts}"):
        return f"single-model-v2-{ts}"
    else:
        return f"single-model-{ts}"

def gen_multi_submission(tss, config, TEST_ON_TRAINING=True, ensemble_func = average_ensemble, all_folds=False, 
                         use_embed=False, save_training_features=False, calibrate_rms=None, gen_file=True,
                         fit_ensemble_preds=None):
    
    
    #pred_sum = np.zeros((len(test)))
    ts_to_scores = {}
    ts_fold_to_scores = {}
    ts_fold_to_embeds = {}
    ts_to_scores_on_training = {}
    ts_fold_to_scores_on_training = {}
    ts_fold_to_embeds_on_training = {}
    for ts in tss:
        upload_dir = get_upload_dir(ts)
        if in_private_env:
            
            config = json.load(open(f'crp/data/{upload_dir}/model_2/training_config.json'))
        else:
            config = json.load(open(f'../input/{upload_dir}/model_2/training_config.json'))

        
        if TEST_ON_TRAINING:
            # test first...
            pred_sum_train, scores_train, embeds_train = pred_df(train[['excerpt','id']], config, 
                    upload_name=get_upload_dir(ts))


            loss_on_train = loss_fn(torch.tensor(pred_sum_train), torch.tensor(train['target'].values), config, loss_type='sqrt_mse').item()
            assert loss_on_train < 0.55

            print(f"loss on training: {loss_on_train}")
            ts_to_scores_on_training[ts] = pred_sum_train
            for i, score in enumerate(scores_train):
                ts_fold_to_scores_on_training[(ts,i)] = score
            for i, embed in enumerate(embeds_train):
                ts_fold_to_embeds_on_training[(ts,i)] = embed
            
        pred, scores, embeds = pred_df(test, config, 
                                       upload_name=get_upload_dir(ts))


        ts_to_scores[ts] = pred
        for i, score in enumerate(scores):
            ts_fold_to_scores[(ts,i)] = score
        for i, embed in enumerate(embeds):
            ts_fold_to_embeds[(ts,i)] = embed
        #pred_sum += pred
        #print(pred.head())
        
        
        torch.cuda.empty_cache()
        gc.collect()
    
    if TEST_ON_TRAINING:
        if use_embed:
            pred_sum_train = ensemble_func(ts_fold_to_embeds_on_training)
            if save_training_features:
                with open('training_features.pkl', 'wb') as f:
                    pickle.dump(ts_fold_to_embeds_on_training, f)
                
        elif all_folds:
            pred_sum_train = ensemble_func(ts_fold_to_scores_on_training)
        else:
            pred_sum_train = ensemble_func(ts_to_scores_on_training)
            
        loss_on_train = loss_fn(torch.tensor(pred_sum_train), torch.tensor(train['target'].values), config, loss_type='sqrt_mse').item()
        print(f"ensembled loss on training: {loss_on_train}")
        assert loss_on_train < 0.55, f"ensembled loss on training: {loss_on_train}"


    #pred_sum = pred_sum/(len(tss))
    if use_embed:
        pred_sum = ensemble_func(ts_fold_to_embeds)
    elif all_folds:
        pred_sum = ensemble_func(ts_fold_to_scores)
    elif fit_ensemble_preds is not None:
        pred_sum = fit_ensemble_preds_then_ensemble(fit_ensemble_preds, ts_to_scores)
    else:
        pred_sum = ensemble_func(ts_to_scores)
    if gen_file:
        pred_res = create_submission(test, pred_sum, calibrate_rms=calibrate_rms)
        pred_res.to_csv('./submission.csv',index=False)
    return pred_sum
    


# In[ ]:





# In[ ]:




# conf utils...

import copy
