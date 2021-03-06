#!/usr/bin/env python
# coding: utf-8



import collections
import torch
#import torchvision
import numpy as np
#import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
#from torchvision.transforms import ToTensor
#from torchvision.utils import make_grid
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
sys.path.append('./')

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
        feature['question'] = example['question']
        feature['answer_text'] = example["answer_text"]

        input_ids = tokenized_example["input_ids"][i]
        attention_mask = tokenized_example["attention_mask"][i]

        feature['input_ids'] = input_ids
        feature['attention_mask'] = attention_mask
        feature['offset_mapping'] = offsets
        feature["example_id"] = example['id']
        feature['sequence_ids'] = [0 if i is None else i for i in tokenized_example.sequence_ids(i)]

        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_example.sequence_ids(i)

        sample_index = sample_mapping[i]
        answers = example["answer_text"]

        if config['USE_CHAR_MODEL'] is not None:
            feature["start_position"] = example["answer_start"]
            feature["end_position"] = start_char + len(example["answer_text"])

        elif False: #len(example["answer_start"]) == 0:
            feature["start_position"] = cls_index
            feature["end_position"] = cls_index
        else:
            start_char = example["answer_start"]
            end_char = start_char + len(example["answer_text"])

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


def prepare_test_features(config, example, tokenizer):
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

    features = []
    for i in range(len(tokenized_example["input_ids"])):
        feature = {}
        feature["example_id"] = example['id']
        feature['context'] = example['context']
        feature['question'] = example['question']
        feature['input_ids'] = tokenized_example['input_ids'][i]
        feature['attention_mask'] = tokenized_example['attention_mask'][i]
        feature['offset_mapping'] = tokenized_example['offset_mapping'][i]
        feature['sequence_ids'] = [0 if i is None else i for i in tokenized_example.sequence_ids(i)]
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
        '''
        if self.mode == 'train':
            return {
                'input_ids': torch.tensor(feature['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(feature['attention_mask'], dtype=torch.long),
                'offset_mapping': torch.tensor(feature['offset_mapping'], dtype=torch.long),
                'start_position': torch.tensor(feature['start_position'], dtype=torch.long),
                'end_position': torch.tensor(feature['end_position'], dtype=torch.long),
                'context': feature['context']
            }
        else:'''
        to_ret = {
            'input_ids': torch.tensor(feature['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(feature['attention_mask'], dtype=torch.long),
            'offset_mapping': torch.tensor(feature['offset_mapping'], dtype=torch.long),
            'sequence_ids': torch.tensor(feature['sequence_ids'], dtype=torch.long),
            'id': feature['example_id'],
            'context': feature['context'],
            'question': feature['question'],
            #'features_index':item
        }
        if 'start_position' in feature:
            to_ret['start_position'] = torch.tensor(feature['start_position'], dtype=torch.long)
            to_ret['end_position'] = torch.tensor(feature['end_position'], dtype=torch.long)
            to_ret['answer_text'] = feature['answer_text']
        return to_ret



def get_stratified_col(train):
    #return train['context'].apply(lambda x: len(x))
    return train['language']

def get_train_and_test_df(root_path='../input/'):
    input_path = root_path

    print(f'loading {input_path}chaii-hindi-and-tamil-question-answering/train.csv')

    train = pd.read_csv(f'{input_path}chaii-hindi-and-tamil-question-answering/train.csv')
    test = pd.read_csv(f'{input_path}chaii-hindi-and-tamil-question-answering/test.csv')
    return train, test



def get_datasets(config, datasets):
    input_path = config['DATA_ROOT_PATH']
    to_ret = []
    for d in datasets:
        if d =='MLQA': #, "XQUAD","QUOREF","NEWSQA"
            external_mlqa = pd.read_csv(f'{input_path}mlqa-hindi-processed/mlqa_hindi.csv')
            if 'id' not in external_mlqa.columns:
                external_mlqa['id'] = [f"mlqa_{x}" for x in external_mlqa.index.values]

            assert 'language' in external_mlqa.columns
            to_ret.append(external_mlqa)
        elif d =='XQUAD': #, "XQUAD","QUOREF","NEWSQA"
            external_xquad = pd.read_csv(f'{input_path}mlqa-hindi-processed/xquad.csv')
            if 'id' not in external_xquad.columns:
                external_xquad['id'] = [f"xquad_{x}" for x in external_xquad.index.values]

            assert 'language' in external_xquad.columns
            to_ret.append(external_xquad)
        elif d =='QUOREF': #, "XQUAD","QUOREF","NEWSQA"
            external_quoref = pd.read_csv(f'{input_path}quoref/quoref_tamil_formated.csv')
            external_quoref['language'] = 'tamil'
            assert 'id' in external_quoref.columns
            to_ret.append(external_quoref)
        elif d =='NEWSQA': #, "XQUAD","QUOREF","NEWSQA"
            external_newsqa = pd.read_csv(f'{input_path}quoref/newsqa_tamil_formated.csv')
            external_newsqa['language'] = 'tamil'
            assert 'id' in external_newsqa.columns
            to_ret.append(external_newsqa)

    to_ret = pd.concat(to_ret)

    return to_ret



def get_data_kfold_split(config):

    input_path = config['DATA_ROOT_PATH']

    train = pd.read_csv(f'{input_path}chaii-hindi-and-tamil-question-answering/train.csv')

    if config['USE_CHAR_MODEL']:
        train = pd.read_pickle(config['CHAR_PROBS_FILE'])

    test = pd.read_csv(f'{input_path}chaii-hindi-and-tamil-question-answering/test.csv')


    external_mlqa = pd.read_csv(f'{input_path}mlqa-hindi-processed/mlqa_hindi.csv')
    # add id:
    external_mlqa['id'] = [f"mlqa_{x}" for x in external_mlqa.index.values]
    external_xquad = pd.read_csv(f'{input_path}mlqa-hindi-processed/xquad.csv')
    # add id:
    external_xquad['id'] = [f"xquad_{x}" for x in external_xquad.index.values]
    external_train = pd.concat([external_mlqa, external_xquad])
    # add quoref:
    if config['USE_QUOREF']:
        external_quoref = pd.read_csv(f'{input_path}quoref/quoref_tamil_formated.csv')
        external_quoref['language'] = 'tamil'
        external_train = external_train.append(external_quoref)


    if config['USE_SIM_SAMPLE'] is not None:
        external_train = pd.read_csv(f'{input_path}sim_sample_cache/{config["USE_SIM_SAMPLE"]}.csv')
        import pickle
        # assert sim sample config match cache...

        with open(f'{input_path}sim_sample_cache/{config["USE_SIM_SAMPLE"]}.pickle', 'rb') as cf:
            cache_config = pickle.load(cf)

        for k,v in cache_config.items():
            assert v == config[k], f"sim sample config key {k}:{v} should match current config:{config[k]}"

        # calculate coverage:
        df_sim_sample_from = get_datasets(config, config['SIM_SAMPLE_DATASETS'])
        logging.info(f"using similarity cache: {config['USE_SIM_SAMPLE']}")
        logging.info(f"similarity sampling covers: {df_sim_sample_from.index.isin(external_train.index.values).mean()}")

    # if USE_CHAR_MODEL, we only use train as we try to learn how they determine char positions
    if config['USE_CHAR_MODEL']:
        external_train = external_train.sample(n=0)


    if config['TEST_RUN']:
        train = train.sample(n=100)
        if not config['USE_CHAR_MODEL']:
            external_train = external_train.sample(n=100)
        logging.info(f"!!! test run !!! n=100")

    if config['STRATEFIED']:
        kfold = StratifiedKFold(n_splits=config['FOLDS'],
                                shuffle=True,
                                random_state=config['SEED'])
    else:
        kfold = KFold(n_splits=config['FOLDS'],
                      random_state=config['SEED'],
                      shuffle=True)

    if config['USE_TRAIN_AS_TEST'] and not config['USE_CHAR_MODEL']:
        external_len = len(external_train)
        train_len = len(train)
        train = external_train.append(train).reset_index(drop=True)

        train_idx = train.iloc[:external_len].index.values
        test_idx = train.iloc[external_len:].index.values
        split_output = [(train_idx, test_idx)]
    else:

        external_len = len(external_train)
        train_len = len(train)
        appended_train = train.append(external_train).reset_index(drop=True)


        if config['STRATEFIED']:
            bins = get_stratified_col(appended_train[:train_len])

            split_output = kfold.split(X=appended_train[:train_len],y=bins)
        else:
            split_output = kfold.split(appended_train[:train_len])

        split_output = [(list(x[0]) + appended_train[train_len:].index.tolist(),x[1]) for x in split_output]
        train = appended_train


    #logging.info(f"data to return: {train}- {split_output}")
    return train, split_output


def optimal_num_of_loader_workers(config):
    if config['USE_CHAR_MODEL'] is not None:
        return 1
    if config['FORCE_NUM_LOADER_WORKER'] is not None:
        num_woker = config['FORCE_NUM_LOADER_WORKER']
        logging.info(f"using number of loader worker: {num_woker}")
        return config['FORCE_NUM_LOADER_WORKER']
    num_cpus = multiprocessing.cpu_count()
    num_gpus = torch.cuda.device_count()
    optimal_value = min(num_cpus, num_gpus*4) if num_gpus else num_cpus - 1
    return optimal_value

def make_loader(
        config,
        data, split_output,
        tokenizer, fold
):
    train_set, valid_set = data.loc[split_output[fold][0]], data.loc[split_output[fold][1]]


    # add augment...
    add_augment_conf = config['ADD_AUGMENT']

    if add_augment_conf is not None and config['AUGMENT_SKIP_TRAINING']:
        train_set = train_set.drop(train_set.index.values)

    if add_augment_conf is not None and isinstance(add_augment_conf, str):
        train_aug = pd.read_csv(add_augment_conf)
        # exclude ids in val:
        logging.info(f"before exclude ids in val len train_aug: {len(train_aug)}")
        train_aug = train_aug[~train_aug.id.isin(valid_set.id.values)]
        logging.info(f"after exclude ids in val len train_aug: {len(train_aug)}")
        train_set = train_set.append(train_aug).sample(frac=1).reset_index(drop=True)
    elif add_augment_conf is not None and isinstance(add_augment_conf, list):
        for aug in add_augment_conf:
            train_aug = pd.read_csv(aug)
            # exclude ids in val:
            logging.info(f"before exclude ids in val len train_aug: {len(train_aug)}")
            train_aug = train_aug[~train_aug.id.isin(valid_set.id.values)]
            logging.info(f"after exclude ids in val len train_aug: {len(train_aug)}")
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
    logging.info(f"Num examples Train= {len(train_dataset)}, Num examples Valid={len(valid_dataset)}")

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
        num_workers=optimal_num_of_loader_workers(config),
        pin_memory=True,
        drop_last=False
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config['VALID_BATCH_SIZE'],
        sampler=valid_sampler,
        num_workers=optimal_num_of_loader_workers(config),
        pin_memory=True,
        drop_last=False
    )
    logging.info(f"loaders created, num steps: train-{len(train_dataloader)}, val-{len(valid_dataloader)}")
    return train_dataloader, valid_dataloader, train_features, valid_features

def make_test_loader(
        config,
        tokenizer,
        df=None):

    input_path = config['DATA_ROOT_PATH']

    if df is None:
        test = pd.read_csv(f'{input_path}/chaii-hindi-and-tamil-question-answering/test.csv')
    else:
        test = df

    logging.info(f"test shape for make test loader: {test.shape}")

    test_features = []
    for i, row in test.iterrows():
        test_features += prepare_test_features(config, row, tokenizer)

    test_dataset = DatasetRetriever(test_features, mode='test')
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['VALID_BATCH_SIZE'],
        sampler=SequentialSampler(test_dataset),
        num_workers=optimal_num_of_loader_workers(config),
        pin_memory=True,
        drop_last=False
    )
    return test_dataloader, test_features

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
            logging.info(data.head())
        
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
def post_cleanup(context, pred):
    bad_starts = [".", ",", "(", ")", "-", "???", ",", ";"]
    bad_endings = ["...", "-", "(", ")", "???", ",", ";"]

    if pred == "":
        return pred

    while any([pred.startswith(y) for y in bad_starts]):
        pred = pred[1:]
    while any([pred.endswith(y) for y in bad_endings]):
        if pred.endswith("..."):
            pred = pred[:-3]
        else:
            pred = pred[:-1]
    return pred


def postprocess_qa_predictions(tokenizer, features,
                               all_start_logits, all_end_logits,
                               n_best_size=20, max_answer_length=30,
                               return_nbest = False,
                               use_char_model = False):
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        if 'example_id' in feature:
            features_per_example[feature["example_id"]].append(i)
        else:
            assert 'id' in feature
            features_per_example[feature["id"]].append(i)



    predictions = collections.OrderedDict()
    predictions_nbest = {}

    logging.debug(f"Post-processing {len(features_per_example)} example predictions split into {len(features)} features.")

    for example_id, features_indice in features_per_example.items():

        min_null_score = None
        valid_answers = []

        context = features[features_indice[0]]['context'] if len(features_indice) > 0 else None
        for feature_index in features_indice:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]

            if use_char_model:
                sequence_ids = features[feature_index]["input_ids"]
                # fix length exceeding
                #start_logits=np.where(sequence_ids != 0, start_logits, np.min(start_logits))
                #end_logits=np.where(sequence_ids != 0, end_logits, np.min(end_logits))

                #start_logits[len(features[feature_index]["context"]):] = [np.min(start_logits) for x in range(len(start_logits) - len(features[feature_index]["context"]))]
                #end_logits[len(features[feature_index]["context"]):] = [np.min(end_logits) for x in range(len(end_logits) - len(features[feature_index]["context"]))]

                start_char = np.argmax(start_logits[:len(features[feature_index]["context"])])
                end_char = np.argmax(end_logits[:len(features[feature_index]["context"])])

                predictions[example_id] = context[start_char: end_char]
            else:


                sequence_ids = features[feature_index]["sequence_ids"]
                context_index = 1
                logging.debug(f"{example_id} offset_mapping: {features[feature_index]['offset_mapping']}")

                features[feature_index]["offset_mapping"] = [
                    (o if sequence_ids[k] == context_index else None)
                    for k, o in enumerate(features[feature_index]["offset_mapping"])
                ]
                offset_mapping = features[feature_index]["offset_mapping"]
                cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
                feature_null_score = start_logits[cls_index] + end_logits[cls_index]
                if min_null_score is None or min_null_score < feature_null_score:
                    min_null_score = feature_null_score

                start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
                end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        logging.debug(f"for {example_id} considering: {start_index} - {end_index}")
                        logging.debug(f"offset_mapping: {offset_mapping}")
                        if (
                                start_index >= len(offset_mapping)
                                or end_index >= len(offset_mapping)
                                or offset_mapping[start_index] is None
                                or offset_mapping[end_index] is None
                        ):
                            continue
                        # Don't consider answers with a length that is either < 0 or > max_answer_length.
                        if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                            continue
                        start_char = offset_mapping[start_index][0]
                        end_char = offset_mapping[end_index][1]
                        valid_answers.append(
                            {
                                "score": start_logits[start_index] + end_logits[end_index],
                                "text": context[start_char: end_char]
                            }
                        )
                        logging.debug(f"found: {context[start_char: end_char]}")

        if not use_char_model:
            # post cleanup
            for i in range(len(valid_answers)):
                valid_answers[i]['text'] = post_cleanup(context, valid_answers[i]['text'])

            if len(valid_answers) > 0:
                best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
            else:
                best_answer = {"text": "", "score": 0.0}

            predictions[example_id] = best_answer["text"]
            predictions_nbest[example_id] = valid_answers

    if return_nbest:
        return predictions, predictions_nbest
    else:
        return predictions


# test...
if __name__ == "__main__":
    from utils.config import TrainingConfig
    config = TrainingConfig({'DATA_ROOT_PATH':'../../chaii/input/', 'USE_TRAIN_AS_TEST': False})
    train, split_output = get_data_kfold_split(config)
    print(train)
    print(split_output)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('deepset/xlm-roberta-large-squad2')
    train_features = []
    for i, row in train.iterrows():
        exp = tokenizer(
            row["question"],
            row["context"],
            truncation="only_second",
            max_length=config['MAX_LEN'],
            stride=config['STRIDE'],
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        if len(exp['input_ids']) > 4:
            print(f"question: {len(tokenizer(row['question'])['input_ids'])}")
            print(f"context: {len(tokenizer(row['context'])['input_ids'])}")
            #print(f"context: {row['context']}")
            print(f"input_ids: {len(exp['input_ids'])}")
            print(f"input_ids: {[len(x) for x in exp['input_ids']]}")

        ret = prepare_train_features(config, row, tokenizer)
        #f len(ret) > 1:
        #    print(f"row: {row} -> features: {ret}")
        train_features += ret

    #nexp = pd.Series([ft['example_id'] for ft in train_features]).nunique()
    print(f"features per example: {len(train_features)/len(train)}")