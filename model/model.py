#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
#import torchvision
import numpy as np
#import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
#from torchvision.transforms import ToTensor
#from torchvision.utils import make_grid
from torch.utils.data import random_split

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
import configparser
warnings.filterwarnings("ignore")
#get_ipython().magic('matplotlib inline')

import os
import time
import random

from sklearn.svm import SVR
import pickle

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
import lightgbm as lgb
#from xgboost import XGBRegressor
#import xgboost as xgb
#from catboost import CatBoostRegressor, Pool, CatBoost


# In[3]:



import os
# only for debugging, this will block multiple GPU utilization
#os.environ["CUDA_LAUNCH_BLOCKING"]="1"   # see issue #152


# In[4]:


#!pip install transformers
import time

import numpy as np
from scipy.stats import norm
from scipy import stats


from transformers import AutoTokenizer,AutoModelForSequenceClassification,BertModel, AutoModel, AutoConfig, BertTokenizer
from transformers import InputExample, InputFeatures
from transformers.file_utils import WEIGHTS_NAME
from transformers import RobertaConfig, RobertaModel
from torch.optim.optimizer import Optimizer
from transformers import (
    get_cosine_schedule_with_warmup, 
    get_cosine_with_hard_restarts_schedule_with_warmup
)
import os
from utils.utils import logging
def get_tokenizer(config):
    if 'megatron' in config['TOKENIZER']:
        tokenizer = BertTokenizer.from_pretrained(config['TOKENIZER'])
    else:
        tokenizer = AutoTokenizer.from_pretrained(config['TOKENIZER'])

    return tokenizer

def save_training_config(config, save_dir):
    config_tosave = copy.copy(config)
    
    #config_tosave['TOKENIZER'] = str(config['TOKENIZER'])
    import pickle
    with open(f'{save_dir}/training_config.pickle', 'wb') as f:
        pickle.dump(config_tosave.config, f)
    logging.info(f"config saved in {save_dir}/training_config.pickle")


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


# In[13]:

# save model...
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import json
import copy

def model_save_pretrained(
    model,
    parallel_model,
    save_directory: Union[str, os.PathLike],
    save_config: bool = True,
    state_dict: Optional[dict] = None,
    save_function: Callable = torch.save,
    push_to_hub: bool = False,
    **kwargs,
):
    """
    Save a model and its configuration file to a directory, so that it can be re-loaded using the
    `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.

    Arguments:
        save_directory (:obj:`str` or :obj:`os.PathLike`):
            Directory to which to save. Will be created if it doesn't exist.
        save_config (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to save the config of the model. Useful when in distributed training like TPUs and need
            to call this function on all processes. In this case, set :obj:`save_config=True` only on the main
            process to avoid race conditions.
        state_dict (nested dictionary of :obj:`torch.Tensor`):
            The state dictionary of the model to save. Will default to :obj:`self.state_dict()`, but can be used to
            only save parts of the model or if special precautions need to be taken when recovering the state
            dictionary of a model (like when using model parallelism).
        save_function (:obj:`Callable`):
            The function to use to save the state dictionary. Useful on distributed training like TPUs when one
            need to replace :obj:`torch.save` by another method.
        push_to_hub (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to push your model to the Hugging Face model hub after saving it.
        kwargs:
            Additional key word arguments passed along to the
            :meth:`~transformers.file_utils.PushToHubMixin.push_to_hub` method.
    """
    if os.path.isfile(save_directory):
        logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
        return
    os.makedirs(save_directory, exist_ok=True)

    # Only save the model itself if we are using distributed training
    model_to_save = unwrap_model(parallel_model)

    # Attach architecture to the config
    model.roberta.config.architectures = [model.roberta.__class__.__name__]

    # Save the config
    if save_config:
        model.roberta.config.save_pretrained(save_directory)

    # Save the model
    if state_dict is None:
        state_dict = model_to_save.state_dict()

    # Handle the case where some state_dict keys shouldn't be saved
    if model.roberta._keys_to_ignore_on_save is not None:
        state_dict = {k: v for k, v in state_dict.items() if k not in model.roberta._keys_to_ignore_on_save}

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
    save_function(state_dict, output_model_file)

    logging.info(f"Model weights saved in {output_model_file}")


class SpanningQAModel(nn.Module):
    def __init__(self, config, from_pretrain=None):
        super(SpanningQAModel, self).__init__()
        self.config = config
        self.model_config = None

        #if from_pretrain is not None:
        #    self.model_config = AutoConfig.from_pretrained(from_pretrain)
        #elif config["MODEL_CONFIG"] is not None:
        #    self.model_config = AutoConfig.from_pretrained(config["MODEL_CONFIG"])




        if from_pretrain is not None:
            logging.info("load pretrain from automodel")

            self.roberta = AutoModel.from_pretrained(from_pretrain, config = self.model_config)

            logging.info("load pretrain directly from file")
            state_dict = torch.load(os.path.join(from_pretrain, WEIGHTS_NAME), map_location=torch.device('cpu'))
            self.load_state_dict(state_dict, strict=False)
            del state_dict
        else:
            self.roberta = AutoModel.from_pretrained(config['BERT_PATH'], config = self.model_config)

        self.qa_outputs = nn.Linear(self.roberta.config.hidden_size, 2)
        self.dropout = nn.Dropout(self.roberta.config.hidden_dropout_prob)
        self._init_weights(self.qa_outputs)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.roberta.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            # token_type_ids=None
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
        )

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # sequence_output = self.dropout(sequence_output)
        qa_logits = self.qa_outputs(sequence_output)

        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

    def load_checkpoint(self, save_directory):

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)

        checkpoint = torch.load(output_model_file, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint, strict=False)
        del checkpoint

    def save_pretrained(
                self,
                parallel_model,
                save_directory: Union[str, os.PathLike],
                save_config: bool = True,
                state_dict: Optional[dict] = None,
                save_function: Callable = torch.save,
                push_to_hub: bool = False,
                **kwargs,
        ):
            return model_save_pretrained(self,
                                         parallel_model,
                                         save_directory,
                                         save_config,
                                         state_dict,
                                         save_function,
                                         push_to_hub,
                                         **kwargs)
class CRPModel(nn.Module):
    def __init__(self, model_config, from_pretrain=None):
        super(CRPModel, self).__init__()
        self.model_config = model_config
        
        
            
        if from_pretrain is not None:
            logging.info("load pretrain from automodel")
            self.roberta = AutoModel.from_pretrained(from_pretrain)
            
            logging.info("load pretrain directly from file")
            state_dict = torch.load(os.path.join(from_pretrain, WEIGHTS_NAME), map_location=torch.device('cpu'))
            self.load_state_dict(state_dict, strict=False)
            del state_dict
        else:
            self.roberta = AutoModel.from_pretrained(model_config['BERT_PATH'])
            
        self.dropout = nn.Dropout(self.model_config['DROPOUT'])
        self.high_dropout = nn.Dropout(self.model_config["USE_MULTI_SAMPLE_DROPOUT_RATE"])
        
        self.fc = nn.Linear(self.roberta.config.hidden_size, self.model_config['BINS_COUNT'] if self.model_config['LOSS_TYPE'] == 'multi-class' else 1)
        
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)


        if self.model_config["USE_BERT_LAST_N_LAYERS"] == -1:
            n_weights = self.roberta.config.num_hidden_layers
        else:
            n_weights = self.model_config["USE_BERT_LAST_N_LAYERS"] #config.num_hidden_layers + 1

        self.n_layers = n_weights


        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)

        self.multi_layer_dropout = nn.Dropout(0.2)

    def load_checkpoint(self, save_directory):
    
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        
        checkpoint = torch.load(output_model_file, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint,strict=False)
        del checkpoint
        
        
    def forward(self, input_ids, attention_mask):

        outputs = self.roberta(input_ids, attention_mask)
        hs = outputs.last_hidden_state
            
        embed = hs[:, 0, :].squeeze(1)


        if not self.model_config["USE_MULTI_SAMPLE_DROPOUT"]:
            x = self.dropout(embed)
            x = self.fc(x)
        else:
          # multisample dropout (wut): https://arxiv.org/abs/1905.09788
          x = torch.mean(
              torch.stack(
                  [self.fc(self.high_dropout(embed)) for _ in range(self.model_config["USE_MULTI_SAMPLE_DROPOUT_SAMPLE"])],
                  dim=0,
              ),
              dim=0,
          )
        return x, embed
    
            
    def save_pretrained(
        self,
        parallel_model,
        save_directory: Union[str, os.PathLike],
        save_config: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        **kwargs,
    ):
        return model_save_pretrained(self, 
            parallel_model,
            save_directory,
            save_config, 
            state_dict,
            save_function,
            push_to_hub,
            **kwargs)
        

class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_dim, num_targets):
        super().__init__()
        if in_features is None:
            in_features = 768
        if hidden_dim is None:
            hidden_dim = 768
        in_features = int(in_features)
        hidden_dim = int(hidden_dim)
        self.in_features = in_features
        self.middle_features = hidden_dim

        self.W = nn.Linear(in_features, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.out_features = hidden_dim

    def forward(self, features):
        att = torch.tanh(self.W(features))

        score = self.V(att)

        attention_weights = torch.softmax(score, dim=1)

        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector
    
class AttentionHeadedModel(nn.Module):
    def __init__(self, model_config, from_pretrain=None):
        super(AttentionHeadedModel,self).__init__()
        
        self.model_config = model_config
        
        #self.roberta = AutoModel.from_pretrained(model_config['BERT_PATH'])
        
        if from_pretrain is not None:
            logging.info("load pretrain from automodel")
            self.roberta = AutoModel.from_pretrained(from_pretrain)
            
            logging.info("load pretrain directly from file")
            state_dict = torch.load(os.path.join(from_pretrain, WEIGHTS_NAME), map_location=torch.device('cpu'))
            self.load_state_dict(state_dict, strict=False)
            del state_dict
        else:
            self.roberta = AutoModel.from_pretrained(model_config['BERT_PATH'])
            
            
        
        self.head = AttentionHead(self.roberta.config.hidden_size,
                                  self.roberta.config.hidden_size,1)
        
        self.dropout = nn.Dropout(self.model_config['DROPOUT'])
        self.high_dropout = nn.Dropout(self.model_config["USE_MULTI_SAMPLE_DROPOUT_RATE"])
        
        #self.linear = nn.Linear(self.head.out_features,1)
        
        self.fc = nn.Linear(self.head.out_features,  1)
        
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self,**xb):
        x = self.roberta(**xb)[0]
    
        if not self.model_config["USE_MULTI_SAMPLE_DROPOUT"]:
            x = self.dropout(x)
        else:
          # multisample dropout (wut): https://arxiv.org/abs/1905.09788
          x = torch.mean(
              torch.stack(
                  [self.high_dropout(x) for _ in range(self.model_config["USE_MULTI_SAMPLE_DROPOUT_SAMPLE"])],
                  dim=0,
              ),
              dim=0,
          )
            
        embed = self.head(x)
        return self.fc(embed), embed

    def load_checkpoint(self, save_directory):
    
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        
        checkpoint = torch.load(output_model_file, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint,strict=False)
            
    def save_pretrained(
        self,
        parallel_model,
        save_directory: Union[str, os.PathLike],
        save_config: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        **kwargs,
    ):
        return model_save_pretrained(self, 
            parallel_model,
            save_directory,
            save_config, 
            state_dict,
            save_function,
            push_to_hub,
            **kwargs)
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding="same", use_bn=True):
        super().__init__()
        if padding == "same":
            padding = kernel_size // 2 * dilation
        
        if use_bn:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation),
                nn.ReLU(),
            )
                
    def forward(self, x):
        return self.conv(x)

class SimpleCNNHeadedModel(nn.Module):
    def __init__(self, model_config, from_pretrain=None, cnn_dim=64, 
                 kernel_size=3, use_bn=False):
        super(SimpleCNNHeadedModel,self).__init__()
        
        self.model_config = model_config
        
        #self.roberta = AutoModel.from_pretrained(model_config['BERT_PATH'])
        
        if from_pretrain is not None:
            logging.info("load pretrain from automodel")
            self.roberta = AutoModel.from_pretrained(from_pretrain)
            
            logging.info("load pretrain directly from file")
            state_dict = torch.load(os.path.join(from_pretrain, WEIGHTS_NAME), map_location=torch.device('cpu'))
            self.load_state_dict(state_dict, strict=False)
            del state_dict
        else:
            self.roberta = AutoModel.from_pretrained(model_config['BERT_PATH'])
            
        self.cnnhead = nn.Sequential(
            ConvBlock(self.roberta.config.hidden_size, cnn_dim, kernel_size=kernel_size, use_bn=use_bn),
            ConvBlock(cnn_dim, cnn_dim * 2, kernel_size=kernel_size, use_bn=use_bn),
            ConvBlock(cnn_dim * 2 , cnn_dim * 4, kernel_size=kernel_size, use_bn=use_bn),
            ConvBlock(cnn_dim * 4, cnn_dim * 8, kernel_size=kernel_size, use_bn=use_bn),
        )
        
        self.attentionhead = AttentionHead(cnn_dim * 8,
                                  self.model_config['attention_dim'],1)
        
        self.dropout = nn.Dropout(self.model_config['DROPOUT'])
        self.high_dropout = nn.Dropout(self.model_config["USE_MULTI_SAMPLE_DROPOUT_RATE"])
    
        self.fc = nn.Linear(cnn_dim * 8,  1)
        
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self,**xb):
        x = self.roberta(**xb)[0]            
        #x = x[:, 0, :]#.squeeze(1)
        
        if not self.model_config["USE_MULTI_SAMPLE_DROPOUT"]:
            x = self.dropout(x)
        else:
          # multisample dropout (wut): https://arxiv.org/abs/1905.09788
          x = torch.mean(
              torch.stack(
                  [self.high_dropout(x) for _ in range(self.model_config["USE_MULTI_SAMPLE_DROPOUT_SAMPLE"])],
                  dim=0,
              ),
              dim=0,
          )
        
        x = x.permute(0, 2, 1)
        x = self.cnnhead(x).permute(0, 2, 1)
        embed = self.attentionhead(x)
        return self.fc(embed), embed

    def load_checkpoint(self, save_directory):
    
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        
        checkpoint = torch.load(output_model_file, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint,strict=False)
            
    def save_pretrained(
        self,
        parallel_model,
        save_directory: Union[str, os.PathLike],
        save_config: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        **kwargs,
    ):
        return model_save_pretrained(self, 
            parallel_model,
            save_directory,
            save_config, 
            state_dict,
            save_function,
            push_to_hub,
            **kwargs)
    
    
# Text CNN model
class textCNN(nn.Module):
    
    def __init__(self, emb_dim, dim_channel, kernel_wins, dropout_rate, num_class):
        super(textCNN, self).__init__()
    
        #Convolutional Layers with different window size kernels
        self.convs = nn.ModuleList([nn.Conv2d(1, dim_channel, (w, emb_dim)) for w in kernel_wins])
        #Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        #FC layer
        self.fc = nn.Linear(len(kernel_wins)*dim_channel, num_class)
        
    def forward(self, x):
        x = x.unsqueeze(1)

        con_x = [conv(x) for conv in self.convs]

        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]
        
        fc_x = torch.cat(pool_x, dim=1)
        
        fc_x = fc_x.squeeze(-1)

        fc_x = self.dropout(fc_x)
        logit = self.fc(fc_x)
        return logit
    
class TextCNNHeadedModel(nn.Module):
    def __init__(self, model_config, from_pretrain=None):
        super(TextCNNHeadedModel,self).__init__()
        self.model_config = model_config
        
        #self.roberta = AutoModel.from_pretrained(model_config['BERT_PATH'])
        
        if from_pretrain is not None:
            logging.info("load pretrain from automodel")
            self.roberta = AutoModel.from_pretrained(from_pretrain)
            
            logging.info("load pretrain directly from file")
            state_dict = torch.load(os.path.join(from_pretrain, WEIGHTS_NAME), map_location=torch.device('cpu'))
            self.load_state_dict(state_dict, strict=False)
            del state_dict
        else:
            self.roberta = AutoModel.from_pretrained(model_config['BERT_PATH'])
        
        self.head = textCNN(self.roberta.config.hidden_size, 100, [3, 4 , 5] , self.model_config['DROPOUT'], 1)
        
        self.dropout = nn.Dropout(self.model_config['DROPOUT'])
        self.high_dropout = nn.Dropout(self.model_config["USE_MULTI_SAMPLE_DROPOUT_RATE"])
                

    def forward(self,**xb):
        x = self.roberta(**xb)[0]
    
        if not self.model_config["USE_MULTI_SAMPLE_DROPOUT"]:
            x = self.dropout(x)
        else:
          # multisample dropout (wut): https://arxiv.org/abs/1905.09788
          x = torch.mean(
              torch.stack(
                  [self.high_dropout(x) for _ in range(self.model_config["USE_MULTI_SAMPLE_DROPOUT_SAMPLE"])],
                  dim=0,
              ),
              dim=0,
          )
        
        embed = x
        x = self.head(embed)
        return x, embed

    def load_checkpoint(self, save_directory):
    
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        
        checkpoint = torch.load(output_model_file, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint,strict=False)
            
    def save_pretrained(
        self,
        parallel_model,
        save_directory: Union[str, os.PathLike],
        save_config: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        **kwargs,
    ):
        return model_save_pretrained(self, 
            parallel_model,
            save_directory,
            save_config, 
            state_dict,
            save_function,
            push_to_hub,
            **kwargs)
    
    
def gbm_scores_ensemble(ts_to_scores, model_save_file, all_folds=False, use_embed=False):
    #load from model:
    
    if use_embed:
        feature_df = np.concatenate([v for (ts, fold),v in ts_to_scores.items()], axis=1)
        logging.info(f"feature cols: {[(ts, fold) for (ts, fold),v in ts_to_scores.items()]}")
    elif not all_folds:
        feature_df = pd.DataFrame({f"pred_{ts}":v for ts,v in ts_to_scores.items()})
    else:
        feature_df = pd.DataFrame({f"pred_{ts}_{fold}":v for (ts, fold),v in ts_to_scores.items()})

    bst = lgb.Booster(model_file=model_save_file)
    res = bst.predict(feature_df)

    return res  

def svr_scores_ensemble(ts_to_scores, model_save_file, all_folds=False, use_embed=False):
    #load from model:
    
    if use_embed:
        feature_df = np.concatenate([v for (ts, fold),v in ts_to_scores.items()], axis=1)
        logging.info(f"feature cols: {[(ts, fold) for (ts, fold),v in ts_to_scores.items()]}")
    elif not all_folds:
        feature_df = pd.DataFrame({f"pred_{ts}":v for ts,v in ts_to_scores.items()})
    else:
        feature_df = pd.DataFrame({f"pred_{ts}_{fold}":v for (ts, fold),v in ts_to_scores.items()})

        
    
    svr_model = SVR(C=10,kernel='rbf',gamma='auto')
    with open(model_save_file, 'rb') as f:
        svr_model = pickle.load(f)

    res = svr_model.predict(feature_df)

    return res


class LitModel(nn.Module):
    def __init__(self, model_config, from_pretrain=None):
        super().__init__()                     
        
        if from_pretrain is not None:
            logging.info("load pretrain from automodel")
            config = AutoConfig.from_pretrained(from_pretrain)
            config.update({"output_hidden_states":True, 
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})  
            
            if model_config['FIX_DROPOUT'] and model_config['ROBERTA_ATTENTION_DROPOUT'] is not None:
                config.update({"attention_probs_dropout_prob": model_config['ROBERTA_ATTENTION_DROPOUT']})
            if model_config['FIX_DROPOUT'] and model_config['ROBERTA_HIDDEN_DROPOUT'] is not None:
                config.update({"hidden_dropout_prob": model_config['ROBERTA_HIDDEN_DROPOUT']})
                
            self.roberta = AutoModel.from_pretrained(from_pretrain, config=config)
            
            logging.info("load pretrain directly from file")
            state_dict = torch.load(os.path.join(from_pretrain, WEIGHTS_NAME), map_location=torch.device('cpu'))
            self.load_state_dict(state_dict, strict=False)
            del state_dict
        else:
            #if 'megatron' not in model_config['BERT_PATH']:
            config = AutoConfig.from_pretrained(model_config['BERT_PATH'])
            config.update({"output_hidden_states":True, 
                           "hidden_dropout_prob": 0.0,
                           "layer_norm_eps": 1e-7})  
            if model_config['FIX_DROPOUT'] and model_config['ROBERTA_ATTENTION_DROPOUT'] is not None:
                config.update({"attention_probs_dropout_prob": model_config['ROBERTA_ATTENTION_DROPOUT']})
            if model_config['FIX_DROPOUT'] and model_config['ROBERTA_HIDDEN_DROPOUT'] is not None:
                config.update({"hidden_dropout_prob": model_config['ROBERTA_HIDDEN_DROPOUT']})
                
            self.roberta = AutoModel.from_pretrained(model_config['BERT_PATH'], config=config)
            #else:
            #    self.roberta = AutoModel.from_pretrained(model_config['BERT_PATH'])
            
        #self.roberta = AutoModel.from_pretrained(ROBERTA_PATH, config=config)  
            
        if model_config['EMBED_OTHER_GPU'] is not None:
            embed_gpu_id = model_config['EMBED_OTHER_GPU']
            self.embeddings = self.roberta.embeddings.to(f'cuda:{embed_gpu_id}')
            self.encoder = self.roberta.encoder.to('cuda:0')
            
        self.attention = nn.Sequential(            
            nn.Linear(self.roberta.config.hidden_size, 512),            
            nn.Tanh(),                       
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        ).to('cuda:0')        

        if model_config['FIX_DROPOUT'] and model_config['HEAD_DROPOUT'] is not None:
            self.head_dropout = nn.Dropout(model_config['HEAD_DROPOUT']).to('cuda:0')
        
        self.regressor = nn.Sequential(                        
            nn.Linear(self.roberta.config.hidden_size, 1)                        
        ).to('cuda:0')
        
        if model_config['FIX_DROPOUT'] and model_config['ROBERTA_ATTENTION_DROPOUT'] is not None:
            logging.info(f"dropout on, dump roberta config:{self.roberta.config}")
        self.model_config = model_config
        

    def forward(self, input_ids, attention_mask):
        
        if self.model_config['EMBED_OTHER_GPU'] is not None:
            embed_gpu_id = self.model_config['EMBED_OTHER_GPU']
            embedding_output = self.embeddings(
                input_ids=input_ids.to(f'cuda:{embed_gpu_id}'),
                mask=attention_mask.to(f'cuda:{embed_gpu_id}'),
            )

            encoder_outputs = self.encoder(
                embedding_output.to('cuda:0'),
                attention_mask.to('cuda:0'),
                output_hidden_states=True,
            )
            last_layer_hidden_states = encoder_outputs.hidden_states[-1]
            
        else:

            roberta_output = self.roberta(input_ids=input_ids,
                                          attention_mask=attention_mask)        


            # There are a total of 13 layers of hidden states.
            # 1 for the embedding layer, and 12 for the 12 Roberta layers.
            # We take the hidden states from the last Roberta layer.
            last_layer_hidden_states = roberta_output.hidden_states[-1]

        # The number of cells is MAX_LEN.
        # The size of the hidden state of each cell is 768 (for roberta-base).
        # In order to condense hidden states of all cells to a context vector,
        # we compute a weighted average of the hidden states of all cells.
        # We compute the weight of each cell, using the attention neural network.
        weights = self.attention(last_layer_hidden_states)
                
        # weights.shape is BATCH_SIZE x MAX_LEN x 1
        # last_layer_hidden_states.shape is BATCH_SIZE x MAX_LEN x 768        
        # Now we compute context_vector as the weighted average.
        # context_vector.shape is BATCH_SIZE x 768
        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)        
        
        # Now we reduce the context vector to the prediction score.
        if self.model_config['FIX_DROPOUT'] and self.model_config['HEAD_DROPOUT'] is not None:
            context_vector = self.head_dropout(context_vector)
        return self.regressor(context_vector), context_vector
    
    def load_checkpoint(self, save_directory):
    
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        
        checkpoint = torch.load(output_model_file, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint,strict=False)
            
            
            
    def save_pretrained(
        self,
        parallel_model,
        save_directory: Union[str, os.PathLike],
        save_config: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        **kwargs,
    ):
        return model_save_pretrained(self, 
            parallel_model,
            save_directory,
            save_config, 
            state_dict,
            save_function,
            push_to_hub,
            **kwargs)
    
class MeanPoolingModel(nn.Module):
    def __init__(self, model_config, from_pretrain=None):
        super().__init__()                     
        
        if from_pretrain is not None:
            logging.info("load pretrain from automodel")
            config = AutoConfig.from_pretrained(from_pretrain)
            self.roberta = AutoModel.from_pretrained(from_pretrain, config=config)
            
            logging.info("load pretrain directly from file")
            state_dict = torch.load(os.path.join(from_pretrain, WEIGHTS_NAME), map_location=torch.device('cpu'))
            self.load_state_dict(state_dict, strict=False)
            del state_dict
        else:

            config = AutoConfig.from_pretrained(model_config['BERT_PATH'])

            self.roberta = AutoModel.from_pretrained(model_config['BERT_PATH'], config=config)
            

        self.linear = nn.Sequential(                        
            nn.Linear(self.roberta.config.hidden_size, 1)                        
        )
        #self.loss = nn.MSELoss()
        
    def forward(self, input_ids, attention_mask, labels=None):
        
        outputs = self.roberta(input_ids, attention_mask)
        last_hidden_state = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        logits = self.linear(mean_embeddings)
        
        return logits, mean_embeddings
    def load_checkpoint(self, save_directory):
    
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        
        checkpoint = torch.load(output_model_file, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint,strict=False)
            
            
            
    def save_pretrained(
        self,
        parallel_model,
        save_directory: Union[str, os.PathLike],
        save_config: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        **kwargs,
    ):
        return model_save_pretrained(self, 
            parallel_model,
            save_directory,
            save_config, 
            state_dict,
            save_function,
            push_to_hub,
            **kwargs)