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

import os
import pickle
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
from utils.config import *

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

from pathlib import Path
import distutils

# for debug...
printed_debug_info = True

def infer(data_loader, model, device, config, tokenizer, use_tqdm=True):
    global printed_debug_info
    printed_debug_info = True
    model.eval()
    to_ret = {}
    with torch.no_grad():
        outputs_starts = None
        outputs_ends = None

        to_for = enumerate(data_loader)
        if use_tqdm:
            to_for = tqdm(to_for, total = len(data_loader))
        for idx, d in to_for:


            model_input_keys = ['input_ids', 'attention_mask']
            data = {key: val.reshape(val.shape[0], -1).to(device) for key, val in d.items() if key in model_input_keys}


            
            if not printed_debug_info:
                logging.info(f"input to model:{data}")
                logging.info(f"input to model shape:{data['attention_mask'].shape}")
                
            outputs_start, outputs_end = model(**data)
            if not printed_debug_info:
                logging.info(f"output to model:{outputs_start}")
                logging.info(f"output to model shape:{outputs_start.shape}")
                logging.info(f"output to model:{outputs_end}")
                logging.info(f"output to model shape:{outputs_end.shape}")
                
                printed_debug_info = True
            outputs_start = outputs_start.squeeze(-1)
            outputs_end = outputs_end.squeeze(-1)

            
            outputs_start = outputs_start.detach().cpu().numpy()
            outputs_end = outputs_end.detach().cpu().numpy()

            if outputs_starts is None:
                outputs_starts = outputs_start
            else:
                outputs_starts = np.concatenate((outputs_starts, outputs_start), axis=0)
            if outputs_ends is None:
                outputs_ends = outputs_end
            else:
                outputs_ends = np.concatenate((outputs_ends, outputs_end), axis=0)


    return outputs_starts, outputs_ends


# In[ ]:

# In[23]:


from sklearn import model_selection,metrics
import numpy as np
import transformers
import pprint
import shutil
import os
import json
from pathlib import Path


    
def pred_df(df, pretrain_base_path, nbest=False, return_logits=False):
    # get config from pretrain path first...
    pretrain_paths = []

    pathlist = Path(pretrain_base_path).glob('**/*')
    for path in pathlist:
        # because path is object not string
        path_in_str = str(path)
        # print(path_in_str)
        if 'model' in path_in_str and os.path.isdir(path_in_str) and path_in_str[-7:-1] == 'model_':
            pretrain_paths.append(path_in_str)
    pretrain_paths.sort()

    if os.path.exists(f'{pretrain_paths[0]}/training_config.pickle'):
        with open(f'{pretrain_paths[0]}/training_config.pickle', "rb") as input_file:
            conf_dict = pickle.load(input_file)

        config = TrainingConfig(conf_dict)
    else:
        config = TrainingConfig(json.load(open(f'{pretrain_paths[0]}/training_config.json')))

    
    if 'SEED' in config:
        seed_everything(config['SEED'])
    else:
        seed_everything(43)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



    pred_sum = np.zeros((len(df)))

    logging.info(f'loadding tokenizer from {pretrain_paths[0]}')

    # fix GPT2 save pretrain issue...
    logging.info(f"loading tokenizer for {config['TOKENIZER']}")
    logging.info(f"loading from {pretrain_paths[0]}/tokenizer/")
    tokenizer = AutoTokenizer.from_pretrained(f"{str(pretrain_paths[0])}/tokenizer/")

    sub_ds_loader,features = make_test_loader(config, tokenizer, df=df)
    
    
    start_logits = None
    end_logits = None

    for p in pretrain_paths:
        model_class = getattr(model_import, config['MODEL_CLASS'])
        
        # hardcode upload fix:
        if os.path.exists(f'{p}/training_config.pickle'):
            with open(f'{p}/training_config.pickle', "rb") as input_file:
                conf_dict = pickle.load(input_file)

            model_config = TrainingConfig(conf_dict)
        else:
            model_config = TrainingConfig(json.load(open(f'{p}/training_config.json')))

        logging.info(f"loading model class:{config['MODEL_CLASS']}\n pretrain: {str(p)}, config:{model_config}")
      
        if model_config['EMBED_OTHER_GPU'] is not None:
            model_config['EMBED_OTHER_GPU'] = 0

            
        model = model_class(from_pretrain=p, config=model_config)
        model.load_checkpoint(p)
        model.to(device)

        pred_start, pred_end = infer(sub_ds_loader,model,device, model_config, tokenizer)

        if start_logits is None:
            start_logits = pred_start
        else:
            start_logits += pred_start
        if end_logits is None:
            end_logits = pred_end
        else:
            end_logits += pred_end
        
        
        # cleanup after fold is done
        logging.info(f'cleanup after model is done')
        del model
        gc.collect()
        torch.cuda.empty_cache()
        

    start_logits = start_logits/(len(pretrain_paths))
    end_logits = end_logits/(len(pretrain_paths))

    if not nbest:
        preds = postprocess_qa_predictions(tokenizer, features,
                                           start_logits.tolist(),
                                           end_logits.tolist(), use_char_model=config['USE_CHAR_MODEL'])
    else:
        preds, preds_nbest = postprocess_qa_predictions(tokenizer, features,
                                           start_logits.tolist(),
                                           end_logits.tolist(), return_nbest=True, use_char_model=config['USE_CHAR_MODEL'])

    df['PredictionString'] = df['id'].map(preds)


    if not nbest:

        ret_df = df
    else:
        df['PredictionStringNBest'] = df['id'].map(preds_nbest)
        ret_df = preds_nbest

    if not return_logits:
        return ret_df
    else:
        return ret_df, start_logits.tolist(), end_logits.tolist(), features
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


def gen_submission(pretrain_base_path, train, test, TRAIN_MODE=False, TEST_ON_TRAINING=True, gen_file=True, nbest=False):
    to_ret = None
    if not TRAIN_MODE:
        if TEST_ON_TRAINING:
            # test first...
            res_df = pred_df(train, pretrain_base_path)
            res_df['jaccard'] = res_df.apply(lambda x: jaccard(x['answer_text'], x['PredictionString']), axis=1)

            jaccard_metric = res_df['jaccard'].mean()
            assert jaccard_metric > 0.8, f"jacarrd on train: {jaccard_metric} should be bigger than 0.8"

            logging.info(f"loss on training: {jaccard_metric}")

        res_df = pred_df(test, pretrain_base_path, nbest)

        pred = res_df[['id', 'PredictionString']]
        logging.info(pred.head())
    else:
        # test infer on training set when it's training mode...
        res_df = pred_df(train, pretrain_base_path)
        res_df['jaccard'] = res_df.apply(lambda x: jaccard(x['answer_text'], x['PredictionString']), axis=1)

        jaccard_metric = res_df['jaccard'].mean()
        assert jaccard_metric > 0.8, f"jacarrd on train: {jaccard_metric} should be bigger than 0.8"

        logging.info(f"loss on training: {jaccard_metric}")

        
    if not TRAIN_MODE and gen_file:
        pred.to_csv('./submission.csv',index=False)

    return pred

def infer_df(df, pretarined_path, nbest):
    return gen_submission(pretarined_path, None, df, False, False, False, nbest)


def get_id_url_from_shared_link(link):
    end = link.index("/view")
    start = link[:end].rindex("/")
    id = link[start+1:end]
    return f"https://drive.google.com/uc?id={id}"

def download_saving(url, saving_ts):
    import gdown
    url = get_id_url_from_shared_link(url)
    #ensure dir:
    os.makedirs("./saved_training", exist_ok=True)
    os.makedirs(f"./saved_training/pretrained-{saving_ts}", exist_ok=True)

    downloaded_file = f"./saved_training/pretrained-{saving_ts}.zip"
    gdown.download(url, downloaded_file, quiet=False)

    import zipfile
    with zipfile.ZipFile(downloaded_file, 'r') as zip_ref:
        zip_ref.extractall(f"./saved_training/pretrained-{saving_ts}")

    os.remove(downloaded_file)


def infer_and_save_inter_outputs(saving_ts, input_base_path, output_base_path, use_train=True, output_logits=True, test_mode=False):
    train, test = get_train_and_test_df(input_base_path)
    str_train = 'train' if use_train else 'test'

    pretrain_base_path = f"{output_base_path}/pretrained-{saving_ts}"

    current_ts = int(time.time())

    os.makedirs(f"{output_base_path}/inter_outputs", exist_ok=True)

    output_path = f"{output_base_path}/inter_outputs/inter_outputs-{str_train}-{current_ts}.pkl"
    #gen_submission(pretrain_base_path, train, test, TRAIN_MODE, TEST_ON_TRAINING, gen_file)

    if test_mode:
        train = train.sample(n=100)

    res_df, start_logits, end_logits, features = pred_df(train if use_train else test,
                                               pretrain_base_path,
                                               return_logits=output_logits)

    if test_mode:
        print(f"logits: {start_logits}, {end_logits}\nfeatures: {features}")

    id_mapping_to_logits = []

    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        if 'example_id' in feature:
            features_per_example[feature["example_id"]].append(i)
        else:
            assert 'id' in feature
            features_per_example[feature["id"]].append(i)

    for i, row in train.iterrows():
        #for example_id, features_indice in features_per_example.items():
        features_indice = features_per_example[row['id']]
        mapping_to_logits = []

        context = features[features_indice[0]]['context'] if len(features_indice) > 0 else None
        for feature_index in features_indice:
            start_logits = start_logits[feature_index]
            end_logits = end_logits[feature_index]
            mapping_to_logits.append((features[feature_index]["offset_mapping"], start_logits, end_logits))
        id_mapping_to_logits.append(mapping_to_logits)

    train['mapping_to_logits'] = id_mapping_to_logits

    train.to_pickle(output_path)
    logging.info(f"saved {output_path}")


def infer_and_gen_submission(saving_ts, base_path, TRAIN_MODE=False, TEST_ON_TRAINING=True, gen_file=True):

    train, test = get_train_and_test_df()
    pretrain_base_path = f"{base_path}/pretrained-{saving_ts}"
    gen_submission(pretrain_base_path, train, test, TRAIN_MODE, TEST_ON_TRAINING, gen_file)

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
            conf_path = f'crp/data/{upload_dir}/model_2/'
        else:
            conf_path = f'../input/{upload_dir}/model_2/'

        if os.path.exists(f'{conf_path}/training_config.pickle'):
            with open(f'{conf_path}/training_config.pickle', "rb") as input_file:
                conf_dict = pickle.load(input_file)

            config = TrainingConfig(conf_dict)
        else:
            config = json.load(open(f'{conf_path}/training_config.json'))

        
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
