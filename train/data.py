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
from torch.utils.data import (random_split,
    Dataset, DataLoader,
    SequentialSampler, RandomSampler
)

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

import multiprocessing

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

def prepare_train_features(config, example, tokenizer):
    example["question"] = example["question"].lstrip()
    tokenized_example = tokenizer(
        example["question"],
        example["context"],
        truncation="only_second",
        max_length=config['MAX_LEN'],
        stride=config['STRIDE'],
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_example.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_example.pop("offset_mapping")

    features = []
    for i, offsets in enumerate(offset_mapping):
        feature = {}
        feature['context'] = example['context']

        input_ids = tokenized_example["input_ids"][i]
        attention_mask = tokenized_example["attention_mask"][i]

        feature['input_ids'] = input_ids
        feature['attention_mask'] = attention_mask
        feature['offset_mapping'] = offsets

        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_example.sequence_ids(i)

        sample_index = sample_mapping[i]
        answers = example["answers"]

        if len(answers["answer_start"]) == 0:
            feature["start_position"] = cls_index
            feature["end_position"] = cls_index
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                feature["start_position"] = cls_index
                feature["end_position"] = cls_index
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                feature["start_position"] = token_start_index - 1
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                feature["end_position"] = token_end_index + 1

        features.append(feature)
    return features


class DatasetRetriever(Dataset):
    def __init__(self, features, mode='train'):
        super(DatasetRetriever, self).__init__()
        self.features = features
        self.mode = mode

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        feature = self.features[item]
        if self.mode == 'train':
            return {
                'input_ids': torch.tensor(feature['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(feature['attention_mask'], dtype=torch.long),
                'offset_mapping': torch.tensor(feature['offset_mapping'], dtype=torch.long),
                'start_position': torch.tensor(feature['start_position'], dtype=torch.long),
                'end_position': torch.tensor(feature['end_position'], dtype=torch.long),
                'context': feature['context']
            }
        else:
            return {
                'input_ids': torch.tensor(feature['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(feature['attention_mask'], dtype=torch.long),
                'offset_mapping': feature['offset_mapping'],
                'sequence_ids': feature['sequence_ids'],
                'id': feature['example_id'],
                'context': feature['context'],
                'question': feature['question']
            }



def get_stratified_col(train):
    return train['context'].apply(lambda x: len(x))

def get_data_kfold_split(config):

    input_path = config['DATA_ROOT_PATH']

    train = pd.read_csv(f'{input_path}chaii-hindi-and-tamil-question-answering/train.csv')
    test = pd.read_csv(f'{input_path}chaii-hindi-and-tamil-question-answering/test.csv')
    external_mlqa = pd.read_csv(f'{input_path}mlqa-hindi-processed/mlqa_hindi.csv')
    external_xquad = pd.read_csv(f'{input_path}mlqa-hindi-processed/xquad.csv')
    external_train = pd.concat([external_mlqa, external_xquad])

    if config['STRATEFIED']:
        kfold = StratifiedKFold(n_splits=config['FOLDS'],
                                shuffle=True,
                                random_state=config['SEED'])
    else:
        kfold = KFold(n_splits=config['FOLDS'],
                      random_state=config['SEED'],
                      shuffle=True)

    if config['USE_TRAIN_AS_TEST']:
        original_train = train
        test = train
        train = external_train
        train = train.append(test)
        train_idx = train[:len(external_train)].id.values
        test_idx = train[len(external_train):].id.values
        split_output = [(train_idx, test_idx)]
    else:


        if config['STRATEFIED']:
            bins = get_stratified_col(train)

            split_output = kfold.split(X=train,y=bins)
        else:
            split_output = kfold.split(train)

    return train, split_output


def optimal_num_of_loader_workers():
    num_cpus = multiprocessing.cpu_count()
    num_gpus = torch.cuda.device_count()
    optimal_value = min(num_cpus, num_gpus*4) if num_gpus else num_cpus - 1
    return optimal_value

def make_loader(
        config,
        data, split_output,
        tokenizer, fold
):






    train_set, valid_set = data.loc[split_output[fold](0)], data.loc[split_output[fold](1)]


    # add augment...
    add_augment_conf = config['ADD_AUGMENT']

    if add_augment_conf is not None and config['AUGMENT_SKIP_TRAINING']:
        train_set = train_set.drop(train_set.index.values)

    if add_augment_conf is not None and isinstance(add_augment_conf, str):
        train_aug = pd.read_csv(add_augment_conf)
        # exclude ids in val:
        print(f"before exclude ids in val len train_aug: {len(train_aug)}")
        train_aug = train_aug[~train_aug.id.isin(valid_set.id.values)]
        print(f"after exclude ids in val len train_aug: {len(train_aug)}")
        train_set = train_set.append(train_aug).sample(frac=1).reset_index(drop=True)
    elif add_augment_conf is not None and isinstance(add_augment_conf, list):
        for aug in add_augment_conf:
            train_aug = pd.read_csv(aug)
            # exclude ids in val:
            print(f"before exclude ids in val len train_aug: {len(train_aug)}")
            train_aug = train_aug[~train_aug.id.isin(valid_set.id.values)]
            print(f"after exclude ids in val len train_aug: {len(train_aug)}")
            train_set = train_set.append(train_aug).sample(frac=1).reset_index(drop=True)

    else:
        train_set = train_set.reset_index(drop=True)

        if config['AUGMENT_REWEIGHTING'] and \
                add_augment_conf is not None:
            id_reweighting_df = train_set.groupby('id', as_index=False).agg(
                reweighting=pd.NamedAgg(column="excerpt", aggfunc="count"))
            train_set = train_set.merge(id_reweighting_df, on='id', how='left')
            train_set['reweighting'] = 1. / train_set['reweighting']

            assert train_set.groupby('id').agg(reweighting_sum=pd.NamedAgg(column='reweighting', aggfunc='sum'))[
                'reweighting_sum'].apply(lambda x: np.isclose(x, 1.0)).all()

    train_features, valid_features = [[] for _ in range(2)]
    for i, row in train_set.iterrows():
        train_features += prepare_train_features(config, row, tokenizer)
    for i, row in valid_set.iterrows():
        valid_features += prepare_train_features(config, row, tokenizer)

    train_dataset = DatasetRetriever(train_features)
    valid_dataset = DatasetRetriever(valid_features)
    print(f"Num examples Train= {len(train_dataset)}, Num examples Valid={len(valid_dataset)}")

    train_sampler = RandomSampler(train_dataset)
    valid_sampler = SequentialSampler(valid_dataset)


    if config['STRATEFIED_SAMPLER']:
        train_sampler = StratifiedBatchSampler(get_stratified_col(train_dataset),
                                         batch_size=config['TRAIN_BATCH_SIZE'],
                                         shuffle=config['SHUFFLE_TRAIN'],
                                         random_state=config['SEED']
                                         )
    elif not config['SHUFFLE_TRAIN']:
        train_sampler = SequentialSampler(valid_dataset)


    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['TRAIN_BATCH_SIZE'],
        sampler=train_sampler,
        num_workers=optimal_num_of_loader_workers(),
        pin_memory=True,
        drop_last=False
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config['VALID_BATCH_SIZE'],
        sampler=valid_sampler,
        num_workers=optimal_num_of_loader_workers(),
        pin_memory=True,
        drop_last=False
    )

    return train_dataloader, valid_dataloader


class CommonLitDataset(nn.Module):
    def __init__(self, data, tokenizer, config, reweighting=False):
        self.config = config
        max_len = config['MAX_LEN']
        self.excerpt = data['excerpt'].to_numpy()
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.targets = data['target'] if 'target' in data.columns else None

        self.weights = None
        if reweighting and 'reweighting' in data.columns:
            self.weights = data['reweighting']
            
        if config['TOKENIZE_ALL']:
            self.encoded = self.tokenizer(data['excerpt'].to_list(),
                            max_length=max_len,
                            padding=self.config['TOKENIZER_PADDING'],
                            truncation=True)

            
        if config['DEBUG_PRINT']:
            print(data.head())
        
    def __len__(self):
        return len(self.excerpt)
    
    def __getitem__(self,item):
        excerpt = self.excerpt[item]
        if self.config['REMOVE_NEWLINE']:
            excerpt = excerpt.replace('\n', '')
        
        if config['TOKENIZE_ALL']:
            inputs = {'input_ids':torch.tensor(self.encoded['input_ids'][item]),
                      'attention_mask':torch.tensor(self.encoded['attention_mask'][item])
                     }
        else:
            inputs = self.tokenizer(excerpt,
                                max_length=config['MAX_LEN'],
                                padding=self.config['TOKENIZER_PADDING'],
                                truncation=True,
                                return_tensors='pt')
        if self.targets is not None:
            target = torch.tensor(self.targets[item], dtype=torch.float if self.loss_type != 'multi-class' else torch.long)  
            if self.weights is not None:
                            
                weight = torch.tensor(self.weights[item], dtype=torch.float) 

                return inputs,target, weight
            else:
                                
                if self.config['DEBUG_PRINT']:
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

