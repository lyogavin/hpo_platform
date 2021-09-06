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
import git

#!pip install transformers
import time


TRAIN_MODE = True
TEST_MODE = False


# In[5]:


# set up logging:


import sys
sys.path.append('../')

from utils.utils import *
from train.loss import *
from train.data import *
from train.optimizer import *
from model.model import *
from train.metric import *
from model import model
from exp_record_store.exp_record import ExpRecord
from utils.utils import logging
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
    original_model.save_pretrained(paralelled_model, f'{saving_dir}/model_{fold}')
    save_training_config(config, f'{saving_dir}/model_{fold}')
    tokenizer.save_pretrained(f'{saving_dir}/model_{fold}/tokenizer')

    # save config another copy in tokenizer
    original_model.roberta.config.save_pretrained(f'{saving_dir}/model_{fold}/tokenizer')

    # save source files...
    path = Path(os.path.dirname(os.path.abspath(__file__)))
    distutils.dir_util.copy_tree(path.parent.absolute(), f'{saving_dir}/src/')

    logging.info(f"saved in {saving_dir}/model_{fold}")

def train_fn(data_loader, valid_loader, fold, model, optimizer, scheduler, device, config,
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
                    
    for idx, d in enumerate(data_loader):

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

        # update meters
        train_steps_meter.update(d['context'], outputs_start, outputs_end, targets_start, targets_end)
        train_total_meter.update(d['context'], outputs_start, outputs_end, targets_start, targets_end)

        if step >= (last_train_eval_step + train_eval_period) or idx == (len(data_loader) -1):
            # Evaluate the model on train_loader.
            num_steps = step - last_eval_step
            last_train_eval_step = step
            train_steps_metrics, train_steps_is_best, train_steps_last_best = train_steps_meter.get_metrics()
            train_total_metrics, train_total_is_best, train_total_last_best = train_total_meter.get_metrics()
            logging.info(f'@desced step {step} @data step {idx} last lr: {min(last_lr)}-{max(last_lr)}\n'
                         f'Train Loss: {loss.item():.4f} Train Steps metrics(new best:{train_steps_is_best}) : {pprint_metrics(train_steps_metrics)}\n'
                         f'Train Total metrics(new best:{train_total_is_best}) : {pprint_metrics(train_total_metrics)}')
            train_steps_meter.reset()


        
        if step >= (last_eval_step + eval_period) or idx == (len(data_loader) -1):
            # Evaluate the model on val_loader.
            num_steps = step - last_eval_step
            last_eval_step = step

            inter_eval_metrics, is_best, last_best = eval(valid_loader,model,device, config, best_metric)
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
    
    return exp_record, best_metric


# In[22]:


def eval(data_loader, model, device, config, previous_best):
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

            meter.update(d['context'], outputs_start, outputs_end, targets_start, targets_end)

        #sp_cor = 0 #spearmanr(fin_targets, fin_outputs)
        
        # bootstrapping confident interval
        #rmse, low, high = bootstrap_rmse(fin_outputs, fin_targets, low_high=True)

        # get metrics for val:
        metrics, is_best, last_best = meter.get_metrics()
        
    return metrics, is_best, last_best


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
                logging.info(f"input to model:{data}")
                logging.info(f"input to model shape:{data['attention_mask'].shape}")
                
            outputs, embeds = model(**data)
            if not printed_debug_info:
                logging.info(f"output to model:{outputs}")
                logging.info(f"output to model shape:{outputs.shape}")
                
                printed_debug_info = True
            outputs = outputs.squeeze(-1)
            
            #outputs = outputs["logits"].squeeze(-1)
            
            if config['LOSS_TYPE'] == 'multi-class':
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

        train_loader, valid_loader = make_loader(config, data, split_output, tokenizer, fold)
            
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
            rec, best_metric = train_fn(train_loader, valid_loader,fold,
                model,
                optimizer,
                scheduler,
                device,
                config,
                exp_record,
                original_model, tokenizer,
                saving_ts, saving_dir, epoch, best_loss_sum, previous_best=best_metric)
        
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




    
def pred_df(df, config, upload_name='pretrained-model-1621892031'):
    pretrain_base_path = f'crp/data/{upload_name}/' if 'gavin_li' in cwd else f'../input/{upload_name}/'
    
    if 'SEED' in config:
        seed_everything(config['SEED'])
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
    pred_sum = np.zeros((len(df)))

    logging.info(f'loadding tokenizer from {pretrain_paths[0]}')

    # fix GPT2 save pretrain issue...
    logging.info(f"loading tokenizer for {config['TOKENIZER']}")
    if config['TOKENIZER'] == 'microsoft/deberta-base':
        logging.info(f"fix GPT2 save pretrain issue by direct load from huggingface registry...")
        if os.path.exists("../input/deberta-base-uncased-tokenizer"):
            logging.info(f"loading from ../input/deberta-base-uncased-tokenizer")
            tokenizer = DebertaTokenizer.from_pretrained(f"../input/deberta-base-uncased-tokenizer")
        elif os.path.exists(f"{str(pretrain_paths[0])}/tokenizer/"):
            logging.info(f"loading from {pretrain_paths[0]}/tokenizer/")
            tokenizer = DebertaTokenizer.from_pretrained(f"{str(pretrain_paths[0])}/tokenizer/")
        else:
            logging.info(f"loading from {pretrain_paths[0]}")
            tokenizer = DebertaTokenizer.from_pretrained(f"{str(pretrain_paths[0])}")
    else:
        if os.path.exists(f"{str(pretrain_paths[0])}/tokenizer/"):
            if config['TOKENIZER'] in ["deepset/roberta-base-squad2","chkla/roberta-argument","roberta-base", "deepset/roberta-large-squad2"]:
                tmp_ts = int(time.time())
                os.system(f'mkdir tmp_{tmp_ts}')
                tokenizer_dir = f"{str(pretrain_paths[0])}/tokenizer/"
                from_dir = f"{str(pretrain_paths[0])}/"
                os.system(f'cp {tokenizer_dir}/* tmp_{tmp_ts}/')
                os.system(f'cp {from_dir}/config.json tmp_{tmp_ts}/')
                tokenizer = AutoTokenizer.from_pretrained(f"tmp_{tmp_ts}/")
            else:
                logging.info(f"loading from {pretrain_paths[0]}/tokenizer/")
                tokenizer = AutoTokenizer.from_pretrained(f"{str(pretrain_paths[0])}/tokenizer/")
        else:
            logging.info(f"loading from {pretrain_paths[0]}")
            tokenizer = AutoTokenizer.from_pretrained(f"{str(pretrain_paths[0])}")

    sub_ds = CommonLitDataset(df, tokenizer, config)    
    
    sub_ds_loader = torch.utils.data.DataLoader(
            sub_ds,
            batch_size = config['VALID_BATCH_SIZE'],
            num_workers = 2,
            drop_last=False,
        )
    
    
    scores = []
    embeds = []

    for p in pretrain_paths:

        #model = AutoModelForSequenceClassification.from_pretrained(p,num_labels=1)
        model_class = getattr(model, config['MODEL_CLASS'])
        
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
        
        logging.info(f"loading model class:{config['MODEL_CLASS']}\n pretrain: {str(p)}, config:{model_config}")
      
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
        logging.info(outputs)

        pred_sum += outputs
        scores.append(outputs)
        embeds.append(output_embeds)
        
        
        # cleanup after fold is done
        logging.info(f'cleanup after model is done')
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

            logging.info(f"loss on training: {loss_on_train}")

        pred_sum, _, _ = pred_df(test, config)
        
        to_ret = pred_sum


        pred = create_submission(test, pred_sum)
        logging.info(pred.head())
    else:
        # test infer on training set when it's training mode...
        pred_sum_train, _, _ = pred_df(train[['excerpt','id']], config)
        loss_on_train = loss_fn(torch.tensor(pred_sum_train), torch.tensor(train['target'].values), config, loss_type='sqrt_mse').item()
        assert loss_on_train < 0.55, f"{loss_on_train} shoudl be small"

        logging.info(f"loss on training: {loss_on_train}")
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
    logging.info(f"fit best score: {math.sqrt(-clf.best_score_)}")
    
    
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

            logging.info(f"loss on training: {loss_on_train}")
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
        #logging.info(pred.head())
        
        
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
        logging.info(f"ensembled loss on training: {loss_on_train}")
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
