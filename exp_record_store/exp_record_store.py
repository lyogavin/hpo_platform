#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

run_ts = int(time.time())

TRAIN_MODE = True
TEST_MODE = False


# In[5]:


# set up logging:

import logging, sys

logging_file_path = f"crp/training_log_{run_ts}.log"

if os.path.exists("crp"):
    handlers=[
        logging.FileHandler(logging_file_path),
        #logging.StreamHandler(sys.stdout)
    ]
else:
    handlers=[
        #logging.FileHandler(logging_file_path),
        logging.StreamHandler(sys.stdout)
    ]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=handlers
)


# In[2]:


import numpy as np
from scipy.stats import norm
from scipy import stats
# referrencing: https://knowledge-repo.d.musta.ch/post/projects/datau332_recalculating_erf_metrics.kp


# import models...
sys.path.append('crp')


import_file_name = 'models_v12_embed_2gpu'

model_import = __import__(import_file_name)
model_import.logging = logging


import math

def bootstrap_rmse(pred, target, sample_size=0.5, sample_count=100, alpha=0.9, squared=False, low_high=False):
    values = []
    se = torch.nn.MSELoss(reduction='none')(torch.tensor(pred), torch.tensor(target)).detach().cpu().numpy()
    #print(f"bootstrap_rmse input: se:{se}\n, se shape:{se.shape}\n")


    rmse = se.mean()
    if not squared:
        rmse = math.sqrt(rmse)

    if not low_high:
        return (rmse, 0, 0)

    to_sample = pd.Series(se)

    for i in range(sample_count):
        sampled = to_sample.sample(int(sample_size * len(pred)))
        to_append = sampled.mean()

        if not squared:
            to_append = math.sqrt(to_append)

        values.append(to_append)

    # confidence intervals
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(values, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(values, p))

    return (rmse, lower,upper)

# inputs are 2 lists where each record is zero or one in each sample entering the experiment:

def calculate_p_value_mde(control_points_or_not_list, treatment_point_or_not_list):
    # 2. Sums and means for numerator and denominator

    assignments_control = len(control_points_or_not_list)
    assignments_treatment = len(treatment_point_or_not_list)

    if assignments_control == 0 or assignments_treatment == 0:
        return 0., 0., 0.

    sum_control_numerator = control_points_or_not_list.sum()
    sum_treatment_numerator = treatment_point_or_not_list.sum()

    sum_control_denominator = assignments_control # If ratio metric, we would sum the denominator values instead
    sum_treatment_denominator = assignments_treatment # If ratio metric, we would sum the denominator values instead

    mean_control_numerator = sum_control_numerator / assignments_control
    mean_control_denominator = sum_control_denominator / assignments_control
    mean_treatment_numerator = sum_treatment_numerator / assignments_treatment
    mean_treatment_denominator = sum_treatment_denominator / assignments_treatment

    # 3. For sample second moments

    sum_control_numerator2 = np.sum(control_points_or_not_list**2)
    sum_control_denominator2 = assignments_control # If ratio metric, we would sum the denominator^2 values instead
    sum_treatment_numerator2 = np.sum(treatment_point_or_not_list**2)
    sum_treatment_denominator2 = assignments_treatment # If ratio metric, we would sum the denominator^2 values instead

    # 4. Variances of numerator and denominator

    variance_control_numerator = sum_control_numerator2 / assignments_control - (sum_control_numerator / assignments_control)**2
    variance_control_denominator = sum_control_denominator2 / assignments_control - (sum_control_denominator / assignments_control)**2
    variance_treatment_numerator = sum_treatment_numerator2 / assignments_treatment - (sum_treatment_numerator / assignments_treatment)**2
    variance_treatment_denominator = sum_treatment_denominator2 / assignments_treatment - (sum_treatment_denominator / assignments_treatment)**2



    # 5. Cross moments (more info)

    cross_moment_control = mean_control_numerator * mean_control_denominator  # adjust this if getting a ratio metric to cross-sum obtained directly from query
    cross_moment_treatment = mean_treatment_numerator * mean_treatment_denominator  # see above

    # 6. Covariance

    co_var_control = cross_moment_control - mean_control_numerator * mean_control_denominator # 0 bc not ratio metric
    co_var_treatment = cross_moment_treatment - mean_treatment_numerator * mean_treatment_denominator # 0 bc not ratio metric


    # 7. Variances of ratio metrics

    variance_treatment =         mean_treatment_denominator**-2         * (
            variance_treatment_numerator \
            + variance_treatment_denominator * (mean_treatment_numerator / mean_treatment_denominator)**2 \
            - (2 * co_var_treatment * (mean_treatment_numerator /  mean_treatment_denominator))
        )

    variance_control =         mean_control_denominator**-2         * (
            variance_control_numerator \
            + variance_control_denominator * (mean_control_numerator / mean_control_denominator)**2 \
            - (2 * co_var_control * (mean_control_numerator /  mean_control_denominator))
        )


    # 8. Means
    mean_control = mean_control_numerator / mean_control_denominator
    mean_treatment = mean_treatment_numerator / mean_treatment_denominator

    if mean_control == 0.:
        percent_change = 0.
    else:
        percent_change = (mean_treatment - mean_control) / mean_control

    # 9. Standard error, Z-statistic, and p-value

    stderr = ((variance_treatment/assignments_treatment) + (variance_control/assignments_control))**0.5
    zstat = (mean_treatment - mean_control) / stderr if stderr != 0 else 0
    pvalue = 2 * (1 - norm.cdf(abs(zstat)))
    ci_95 = stderr * 1.959967124 / mean_control  if mean_control != 0 else 0

    # 10. Comparing against ground truth

    if False:
        print("sum_control_numerator is {}".format(sum_control_numerator))
        print("sum_control_denominator is {}".format(sum_control_denominator))
        print("sum_treatment_numerator is {}".format(sum_treatment_numerator))
        print("sum_treatment_denominator is {}".format(sum_treatment_denominator))
        print("mean_treatment is {}".format(mean_treatment))
        print("mean_control is {}".format(mean_control))
        print("percent_change is {}".format(percent_change))
        print("pvalue is {}".format(pvalue))
        print("stderr is {}".format(stderr))
        print("ci_95 is {}".format(ci_95))



    # Power:
    # let's work from values we would see on the ERF UI; hint, in terms of
    # scipy stats functions, you might need:
    # norm.isf - returns z critical value on right tail
    # norm.ppf - returns z critical value on  left tail
    # norm.sf  - returns area under curve right of a z critical value
    # norm.cdf - returns area under curve  left of a z critical value

    detectable_lift = 0.73            # MDE on ERF UI
    base_rate       = mean_control    # 45 / 17986
    standard_error  = stderr          # calculated earlier!
    alpha           = 0.05 / 2        # two-tailed

    power    = 0
    delta    = detectable_lift * base_rate

    Zcrit_r  = stats.norm.isf(alpha)                             # +1.96, as expected
    power   += stats.norm.sf( Zcrit_r - delta / standard_error)  if standard_error != 0 else 0 # norm.sf is area to right tail

    Zcrit_l  = stats.norm.ppf(alpha)                             # -1.96, as expected
    power   += stats.norm.cdf(Zcrit_l - delta / standard_error)  if standard_error != 0 else 0 # norm.cdf is area to left tail

    #print("")
    #print("Power corresponding to a 73% MDE is: {:0.2f}".format(power)) # shockingly close eh...

    # MDE

    MDD = 2.80158178701

    mde = MDD * standard_error / mean_control if mean_control != 0 else 0
    #print("")
    #print("MDE is {:0.2f}".format(mde)) # shockingly close eh?!

    return pvalue, mde, percent_change


# In[6]:


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


cwd = os.path.abspath(os.getcwd())
in_private_env = 'gavin_li' in cwd

input_path = "../input/" if not in_private_env else 'crp/input/'






path = 'crp/data/' if in_private_env else '../input/commonlitreadabilityprize/'
train = pd.read_csv(path + 'train.csv')
train = train[(train.target != 0) & (train.standard_error != 0)].reset_index(drop=True)
test = pd.read_csv(path + 'test.csv')
sample = pd.read_csv(path + 'sample_submission.csv')

if TEST_MODE:
    train = train[:1000]

nbins = 12 #config['BINS_COUNT']
train.loc[:,'bins'], cut_bins = pd.cut(train['target'],nbins,labels=False, retbins=True)

bins = train.bins.to_numpy()


# In[9]:


bin_values = train.groupby('bins', as_index=False).agg({'target':'mean'}).sort_values('bins')['target'].values
bin_values


# ## add reverse se for weighting...

# In[10]:


train['reverse_se'] = train['standard_error'].apply(lambda x: 1.0/x if x >0 else None)
train['reverse_se'] = train['reverse_se'].fillna(train['reverse_se'].max())
train['sq_reverse_se'] = train['reverse_se'] * train['reverse_se']


# In[11]:


print(train.head())

print(len(train))


# In[12]:



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


y_train = train['target'].to_numpy()


# In[14]:

DEBUG_PRINT = False


class CommonLitDataset(nn.Module):
    def __init__(self, data, tokenizer, config, reweighting=False):
        self.config = config
        max_len = config['MAX_LEN']
        loss_type=config['LOSS_TYPE']
        self.excerpt = data['excerpt'].to_numpy()
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.targets = data['target'] if 'target' in data.columns else None
        self.loss_type = loss_type
        if loss_type == 'multi-class' and 'bis' in data.columns:
            self.targets = data['bins']
        self.weights = None
        if reweighting and 'reweighting' in data.columns:
            self.weights = data['reweighting']

        if config['TOKENIZE_ALL']:
            self.encoded = self.tokenizer(data['excerpt'].to_list(),
                            max_length=self.config['MAX_LEN'],
                            padding=self.config['TOKENIZER_PADDING'],
                            truncation=True)


        if DEBUG_PRINT:
            print(data.head())

    def __len__(self):
        return len(self.excerpt)

    def __getitem__(self,item):
        excerpt = self.excerpt[item]
        if self.config['REMOVE_NEWLINE']:
            excerpt = excerpt.replace('\n', '')

        if self.config['TOKENIZE_ALL']:
            inputs = {'input_ids':torch.tensor(self.encoded['input_ids'][item]),
                      'attention_mask':torch.tensor(self.encoded['attention_mask'][item])
                     }
        else:
            inputs = self.tokenizer(excerpt,
                                max_length=self.config['MAX_LEN'],
                                padding=self.config['TOKENIZER_PADDING'],
                                truncation=True,
                                return_tensors='pt')
        if self.targets is not None:
            target = torch.tensor(self.targets[item], dtype=torch.float if self.loss_type != 'multi-class' else torch.long)
            if self.weights is not None:

                weight = torch.tensor(self.weights[item], dtype=torch.float)

                return inputs,target, weight
            else:

                if DEBUG_PRINT:
                    return inputs,target, excerpt
                else:
                    return inputs,target


        else:
            return inputs



# In[15]:


def weighted_mse_loss(input, target, weight):
    return torch.sqrt(torch.mean(weight * (input - target) ** 2))

class BerHuLoss(nn.Module):
    """Class implementing the BerHu loss."""
    def __init__(self, threshold=0.2):
        """
        Initializes the BerHuLoss class.
        Parameters
        ----------
        threshold : float
            Mask parameter
        """
        super().__init__()
        self.threshold = threshold
    def forward(self, pred, gt):
        """
        Calculates the BerHu loss.
        Parameters
        ----------
        pred : torch.Tensor [B,1,H,W]
            Predicted inverse depth map
        gt : torch.Tensor [B,1,H,W]
            Ground-truth inverse depth map
        Returns
        -------
        loss : torch.Tensor [1]
            BerHu loss
        """
        huber_c = torch.max(pred - gt)
        huber_c = self.threshold * huber_c
        diff = (pred - gt).abs()

        # Remove
        # mask = (gt > 0).detach()
        # diff = gt - pred
        # diff = diff[mask]
        # diff = diff.abs()

        huber_mask = (diff > huber_c).detach()
        diff2 = diff[huber_mask]
        diff2 = diff2 ** 2
        return torch.cat((diff, diff2)).mean()

def smooth_l1_loss(
    input: torch.Tensor, target: torch.Tensor, beta: float, reduction: str = "none"
) -> torch.Tensor:
    """
    Smooth L1 loss defined in the Fast R-CNN paper as:
    ::
                      | 0.5 * x ** 2 / beta   if abs(x) < beta
        smoothl1(x) = |
                      | abs(x) - 0.5 * beta   otherwise,

    where x = input - target.

    Smooth L1 loss is related to Huber loss, which is defined as:
    ::
                    | 0.5 * x ** 2                  if abs(x) < beta
         huber(x) = |
                    | beta * (abs(x) - 0.5 * beta)  otherwise

    Smooth L1 loss is equal to huber(x) / beta. This leads to the following
    differences:

     - As beta -> 0, Smooth L1 loss converges to L1 loss, while Huber loss
       converges to a constant 0 loss.
     - As beta -> +inf, Smooth L1 converges to a constant 0 loss, while Huber loss
       converges to L2 loss.
     - For Smooth L1 loss, as beta varies, the L1 segment of the loss has a constant
       slope of 1. For Huber loss, the slope of the L1 segment is beta.

    Smooth L1 loss can be seen as exactly L1 loss, but with the abs(x) < beta
    portion replaced with a quadratic function such that at abs(x) = beta, its
    slope is 1. The quadratic segment smooths the L1 loss near x = 0.

    Args:
        input (Tensor): input tensor of any shape
        target (Tensor): target value tensor with the same shape as input
        beta (float): L1 to L2 change point.
            For beta values < 1e-5, L1 loss is computed.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.

    Returns:
        The loss with the reduction option applied.

    Note:
        PyTorch's builtin "Smooth L1 loss" implementation does not actually
        implement Smooth L1 loss, nor does it implement Huber loss. It implements
        the special case of both in which they are equal (beta=1).
        See: https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss.
    """
    if beta < 1e-5:
        # if beta == 0, then torch.where will result in nan gradients when
        # the chain rule is applied due to pytorch implementation details
        # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
        # zeros, rather than "no gradient"). To avoid this issue, we define
        # small values of beta to be exactly l1 loss.
        loss = torch.abs(input - target)
    else:
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def loss_fn(output,target, config, weight=None, loss_type=None):
    if loss_type is None:
        loss_type=config['LOSS_TYPE']

    if loss_type == 'multi-class':

        ce_loss = nn.CrossEntropyLoss()

        return ce_loss(output, target)



    if weight is not None and config['LOSS_TYPE'] == 'weighted_mse':
        return weighted_mse_loss(output, target, weight)
    else:
        if loss_type == 'sqrt_mse':
            return torch.sqrt(nn.MSELoss()(output,target))
        elif loss_type == 'mse':
            return nn.MSELoss()(output,target)
        elif loss_type == 'l1':
            return nn.L1Loss()(output,target)
        elif loss_type == 'smoothl1':
            return smooth_l1_loss(output,target, 1.0, "sum")
        elif loss_type == 'berhu':
            return BerHuLoss()(output,target)


# In[16]:


def seed_everything(seed=43):
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True
    #config['SEED'] = seed


# ## Discriminative Learning Rate and Weight Decay

# In[17]:


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

    for layer_num, (name, params) in enumerate(roberta_parameters):
        weight_decay = 0.0 if "bias" in name else 0.01

        lr = config['LR'] #2e-5

        if layer_num >= roberta_mid_layers[model_bert_path]:
            lr = config['LR'] * 2.5 #5e-5

        if layer_num >= roberta_late_layers[model_bert_path]:
            lr = config['LR'] * 5.0 #1e-4

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

def make_scheduler(optimizer, train_ds, config, train_loader):
    decay_name=config['DECAY_NAME'] #'cosine_warmup',
    #t_max=config['EPOCHS']

    grad_accu_factor = 1

    if config['GRAD_ACCU_STEPS'] is not None:
        grad_accu_factor = config['GRAD_ACCU_STEPS']
    t_max = int(len(train_ds) / config['TRAIN_BATCH_SIZE'] * config['EPOCHS'] / grad_accu_factor)

    if config['FIX_STEPS_BUG']:
        multi_gpu_batch_size = 1

        if config['GPU_PARALLEL_IDS'] is not None:
            multi_gpu_batch_size = len(config['GPU_PARALLEL_IDS'])

        t_max = int(len(train_loader) * config['EPOCHS'] * config['STEPS_FACTOR'])

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
            lr_end=config['POLY_DECAY_LR_END'] * config['LR']
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


# In[20]:


# save model...
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import json
import copy


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
    training_log = config['TRAINING_LOG']

    losses = []

    dumped_data = 0

    if config['FREEZE_EMBED']:
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

        if config['REMOVE_TOKEN_TYPES'] and 'token_type_ids' in data:
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

        if config['AUTO_SCALER']:
            with torch.cuda.amp.autocast():
                outputs, _ = model(**data)
        else:
            outputs, _ = model(**data)
        outputs = outputs.squeeze(-1)
        #Eprint(outputs)

        loss = loss_fn(outputs, targets, config, weights)
        if config['GRAD_ACCU_STEPS'] is not None:
            loss = loss / config['GRAD_ACCU_STEPS']

        losses.append(loss.item())
        loss.backward()

        if config['GRAD_ACCU_STEPS'] is None:
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

            if config['REMOVE_TOKEN_TYPES'] and 'token_type_ids' in data:
                del data['token_type_ids']

            targets = targets.to(device)

            if config['AUTO_SCALER']:
                with torch.cuda.amp.autocast():
                    outputs, _ = model(**data)
            else:
                outputs, _ = model(**data)

            outputs = outputs.squeeze(-1)

            #outputs = outputs["logits"].squeeze(-1)

            if config['LOSS_TYPE'] == 'multi-class':
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

    if 'megatron' in config['TOKENIZER']:
        tokenizer = BertTokenizer.from_pretrained(config['TOKENIZER'])
    else:
        tokenizer = AutoTokenizer.from_pretrained(config['TOKENIZER'])

    if config['STRATEFIED']:
        kfold = StratifiedKFold(n_splits=config['FOLDS'],shuffle=True,random_state=config['SEED'])
    else:
        kfold = KFold(n_splits=config['FOLDS'], random_state=config['SEED'], shuffle=True)

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
    config['TRAINING_LOG'] = training_log

    best_loss_sum = 0.

    if config['STRATEFIED']:
        split_output = kfold.split(X=train,y=bins)
    else:
        split_output = kfold.split(train)

    all_folds_outputs_targets = ([], [])

    for fold , (train_idx,valid_idx) in enumerate(tqdm(split_output, total=config['FOLDS'])):

        if config['NEW_SEEDS']:
            seed_everything(config['SEED'] + fold)
        if config['STOP_AT_FOLD'] == fold:
            logging.info(f"stopping at {fold}...")
            break
        start_time = time.time()
        train_x,valid_x = train.loc[train_idx],train.loc[valid_idx]

        if config['ADD_AUGMENT'] is not None and config['AUGMENT_SKIP_TRAINING']:
            train_x = train_x.drop(train_x.index.values)

        if config['ADD_AUGMENT'] is not None and isinstance(config['ADD_AUGMENT'], str):
            train_aug = pd.read_csv(config['ADD_AUGMENT'])
            # exclude ids in val:
            print(f"before exclude ids in val len train_aug: {len(train_aug)}")
            train_aug = train_aug[~train_aug.id.isin(valid_x.id.values)]
            print(f"after exclude ids in val len train_aug: {len(train_aug)}")
            train_x = train_x.append(train_aug).sample(frac=1).reset_index(drop=True)
        elif config['ADD_AUGMENT'] is not None and isinstance(config['ADD_AUGMENT'], list):
            for aug in config['ADD_AUGMENT']:
                train_aug = pd.read_csv(aug)
                # exclude ids in val:
                print(f"before exclude ids in val len train_aug: {len(train_aug)}")
                train_aug = train_aug[~train_aug.id.isin(valid_x.id.values)]
                print(f"after exclude ids in val len train_aug: {len(train_aug)}")
                train_x = train_x.append(train_aug).sample(frac=1).reset_index(drop=True)

        else:
            train_x = train_x.reset_index(drop=True)

        if config['AUGMENT_REWEIGHTING'] and config['ADD_AUGMENT'] is not None:
            id_reweighting_df = train_x.groupby('id', as_index=False).agg(reweighting=pd.NamedAgg(column="excerpt", aggfunc="count"))
            train_x = train_x.merge(id_reweighting_df, on='id', how='left')
            train_x['reweighting'] = 1. / train_x['reweighting']

            assert train_x.groupby('id').agg(reweighting_sum=pd.NamedAgg(column='reweighting',aggfunc='sum'))['reweighting_sum'].apply(lambda x: np.isclose(x, 1.0)).all()

        valid_x = valid_x.reset_index(drop=True)

        train_ds = CommonLitDataset(train_x, tokenizer, config, config['ADD_AUGMENT'])

        multi_gpu_batch_size = 1

        if config['GPU_PARALLEL_IDS'] is not None:
            multi_gpu_batch_size = len(config['GPU_PARALLEL_IDS'])

        if not config['STRATEFIED_SAMPLER']:
            train_loader = torch.utils.data.DataLoader(
                train_ds,
                batch_size = config['TRAIN_BATCH_SIZE'] * multi_gpu_batch_size,
                num_workers = 2,
                shuffle = config['SHUFFLE_TRAIN'],
                drop_last=True,
            )
        else:
            #y, batch_size, shuffle=True, random_state=42
            sampler = StratifiedBatchSampler(train_x['bins'],
                                             batch_size=config['TRAIN_BATCH_SIZE'] * multi_gpu_batch_size,
                                             shuffle=config['SHUFFLE_TRAIN'],
                                             random_state=config['SEED']
                                            )
            train_loader = torch.utils.data.DataLoader(
                train_ds,
                batch_sampler = sampler,
                num_workers = 2,
            )

        valid_ds = CommonLitDataset(valid_x, tokenizer, config)

        valid_loader = torch.utils.data.DataLoader(
            valid_ds,
            batch_size = config['VALID_BATCH_SIZE'] * multi_gpu_batch_size,
            num_workers = 2,
            drop_last=False,
        )

        if config['NEW_SEEDS']:
            seed_everything(config['SEED'] + fold)

        #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        logging.info(f"training config: {pprint.pformat(config)}")
        logging.info(f"========== USING {device} ==========")
        logging.info(f'========== Fold: {fold} ==========')
        num_labels=1
        if config['LOSS_TYPE'] == 'multi-class':
            num_labels = config['BINS_COUNT']
        #original_model = AutoModelForSequenceClassification.from_pretrained(config['BERT_PATH'],num_labels=num_labels)

        model_class = getattr(model_import, config['MODEL_CLASS'])



        if config['PRETRAIN_TO_LOAD'] is not None:
            original_model = model_class(from_pretrain=config['PRETRAIN_TO_LOAD'], model_config=config)
        else:
            original_model = model_class(model_config=config)



        if config['FIX_DROPOUT'] and config['HEAD_DROPOUT'] is not None:
            assert "(head_dropout): Dropout" in str(original_model)
            #print(f"dropout on, dump model: {original_model}")

        #torch.save(original_model.state_dict(), "/tmp/b0")
        #print(f"model hash saved in /tmp/b0")

        if config['EMBED_OTHER_GPU'] is None:
            original_model.to(device)

        if config['GPU_PARALLEL_IDS'] is not None:
            print(f"using device ids: {config['GPU_PARALLEL_IDS']}")
            logging.info(f"using device ids: {config['GPU_PARALLEL_IDS']}")
            model =  torch.nn.DataParallel(original_model, device_ids=config['GPU_PARALLEL_IDS'])
        else:

            model = original_model

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias','LayerNorm.bias','LayerNorm.weight']
        optimizer_parameters = [
            {'params' : [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay' : 0.001},
            {'params' : [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay' : 0.0},
        ]

        num_train_steps = int(len(train_ds) / config['TRAIN_BATCH_SIZE'] * config['EPOCHS'])

        if config['FIX_STEPS_BUG']:
            grad_accu_factor = 1

            if config['GRAD_ACCU_STEPS'] is not None:
                grad_accu_factor = config['GRAD_ACCU_STEPS']
            num_train_steps = int(len(train_loader) * config['EPOCHS'] * config['STEPS_FACTOR'] / grad_accu_factor)



#         optimizer = AdamW(optimizer_parameters, lr = 3e-5, betas=(0.9, 0.999))
        if config['USE_SIMPLE_OPTIMIZER']:
            optimizer = AdamW(model.parameters(), lr = config['LR'], betas=(0.9, 0.999),
                              weight_decay=config['ADAM_WEIGHT_DECAY']#1e-5
                             )
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps = int(config['WARMUP_STEPS_RATIO'] * num_train_steps),
                num_training_steps = num_train_steps
            )
#         scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=len(train_ds), max_lr=1e-4, epochs=config['EPOCHS'])
        else:
            optimizer = make_optimizer(original_model, config)
            scheduler = make_scheduler(optimizer, train_ds, config, train_loader)

        best_loss = 99999

        losses_valid = list()
        best_preds = list()

        if config['VAL_STEPS_CHUNKS'] is not None:
            num_steps = total_steps // config['VAL_STEPS_CHUNKS']

        if config['SCHEDULED_EVAL'] is not None and config['SCHEDULED_EVAL']:
            EVAL_SCHEDULE = [(0.50, 16), (0.49, 8), (0.48, 4), (0.47, 2), (-1., 1)]
            num_steps = EVAL_SCHEDULE[0][1]


        torch.cuda.empty_cache()

        for epoch in range(config['EPOCHS']):
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


# ## prepare config matrix

# In[24]:


import copy
def generate_config_list(configs_to_explore):
    configs_to_explore = copy.deepcopy(configs_to_explore)

    config_list = []

    found_explore_key = False
    for k,v in configs_to_explore.items():
        if isinstance(v, dict) and 'to_explore' in v:
            if 'on_false' in v and configs_to_explore[v['on_false']]:
                configs_to_explore[k] = v['to_explore'][0]
                continue
            if 'on_true' in v and not configs_to_explore[v['on_true']]:
                configs_to_explore[k] = v['to_explore'][0]
                continue
            #print(f"exploing: {(k,v)}")
            found_explore_key = True
            for i in v['to_explore']:
                configs_to_explore[k] = i
                ret = generate_config_list(configs_to_explore)
                config_list.extend(ret)
            break
        elif isinstance(v, dict) and 'copy_from' in v:
            configs_to_explore[k] = configs_to_explore[v['copy_from']]
        elif isinstance(v, dict) and 'is_group' in v and ['is_group']:
            found_explore_key = True
            for d in v['groups_to_explore']:
                configs_to_explore = {**configs_to_explore, **d}

                to_recur = copy.deepcopy(configs_to_explore)
                del to_recur[k]
                ret = generate_config_list(to_recur)
                config_list.extend(ret)
            break

    if not found_explore_key:
        config_list.append(configs_to_explore)
    return config_list

def constraint_conflist(confs, constraint):
    #print(confs)
    #print(constraint)
    confs = [{**x, **constraint} for x in confs]
    to_ret = []
    for conf in confs:
        if conf not in to_ret:
            to_ret.append(conf)
    return to_ret
# In[ ]:



base_config = {
#'SEED': {'to_explore':[3,13,23,33,42,43,53,133,168,200]},
'MAX_LEN' : 267,
'TRAIN_BATCH_SIZE' : 8, #{'to_explore':[8,16,32]}, #8,
'VALID_BATCH_SIZE' : 8, #4,
'EPOCHS' : 10,
'BERT_PATH' : "roberta-base",
'CSV_PATH' : 'lgbmtrain.csv',
'AUGMENTED_CSV' : 'lgbmtrainAUG.csv',
'MODEL_PATH' : './CLRPmodel',
'TOKENIZER' : 'roberta-base', #AutoTokenizer.from_pretrained('roberta-base') if TRAIN_MODE else None,
'VAL_STEPS_CHUNKS': None,
'NEW_SEEDS':False,

# preprocess:
'REMOVE_NEWLINE': False,
# loss part:
'LOSS_TYPE': 'sqrt_mse', #'weighted_mse', #'multi-class', #
'BINS_COUNT': 12,
'WEIGHT_TYPE': 'reversed_se', #'sq_reversed_se', #
'TOKENIZER_PADDING':"max_length",

# training:
'GPU_PARALLEL_IDS': None, #list(range(8)), #None, #
'FOLDS': 3, #6, #2, #
'STOP_AT_FOLD':3,
'SAVING_STOP_AT_FOLD':6,
'SCHEDULED_EVAL':False,
'SHUFFLE_TRAIN':False,

# drop out:
"USE_MULTI_SAMPLE_DROPOUT": False, #https://github.com/oleg-yaroshevskiy/quest_qa_labeling/blob/master/step5_model3_roberta_code/model.py#L119
'DROPOUT': 0, #0.5, #0, #
"USE_MULTI_SAMPLE_DROPOUT_RATE": 0.75,
"USE_MULTI_SAMPLE_DROPOUT_SAMPLE":  4,
"ROBERTA_HIDDEN_DROPOUT":None,
"ROBERTA_ATTENTION_DROPOUT":None,
"HEAD_DROPOUT":None,
"FIX_DROPOUT":False,

# model arch
"MODEL_CLASS": "CRPModel",
"USE_BERT_ALL_LAYERS": False,
"USE_BERT_LAST_4_LAYERS": False,
#https://github.com/oleg-yaroshevskiy/quest_qa_labeling/blob/master/step5_model3_roberta_code/model.py#L76
"USE_BERT_LAST_N_LAYERS": 4,
"attention_dim":None, #768,
'FREEZE_EMBED':False,
'EMBED_OTHER_GPU':None,


# pretraining
"PRETRAIN_TO_LOAD": f"{input_path}robertaitpt", #

# optimizer
'LR': 4e-5,#4e-5, #
"USE_SIMPLE_OPTIMIZER": True,
"OPTIMIZER_NAME": "AdamW",
'WEIGHT_DECAY':0,
'DECAY_NAME':'cosine_warmup',
'FIX_STEPS_BUG':False,
'WARMUP_STEPS_RATIO':0,
'ADAM_WEIGHT_DECAY':1e-5,
'ADAMW_BETAS':None,
'ADAMW_EPS':None,
'POLY_DECAY_LR_END':1e-7,
"STEPS_FACTOR":1.0,
'NUM_CYCLES':None,
"AUTO_SCALER":False,
"GRAD_ACCU_STEPS":None,
'LAYERED_OPT_DEFAULT_LR':None,
'LAYERED_OPT_DEFAULT_WEIGHT_DECAY':None,
'LAYERED_OPT_ENABLED':'albert-xxlarge-v2,roberta-base',


# data augment:
'ADD_AUGMENT':None, #"crp/data/train_sent_sample_0.5.csv",
'AUGMENT_SKIP_TRAINING':False,
'AUGMENT_REWEIGHTING':False,
'STRATEFIED':True,
'TOKENIZE_ALL':False,
'STRATEFIED_SAMPLER':False,
'POSTREPRO':False,
'EXP_VERSION':1,

'REMOVE_TOKEN_TYPES':False,
}

SEED_to_explore = [3, 13, 23, 33, 42, 43, 53, 133, 168, 200, 75, 62, 90, 57, 72, 11, 54, 70, 61, 67,
               51, 80, 57, 84, 18, 67, 73, 18, 4, 12, 72, 85, 59, 34, 6, 20, 85, 93, 63, 74, 36,
               28, 71, 62, 20, 2, 75, 20, 16, 56, 65, 29, 15, 69, 9, 98, 55, 78, 6, 60, 54, 98,
               34, 31, 36, 24, 69, 44, 98, 80, 34, 2, 98, 55, 80, 27, 41, 39, 20, 82, 22, 32, 56,
               35, 44, 48, 94, 90, 36, 68, 99, 34, 75, 25, 51, 95, 75, 22, 37, 22, 88, 85, 64, 96,
               31, 72, 57, 21, 36, 96, 50, 38, 32, 40, 77, 25, 89, 70, 39, 7, 95, 10, 70, 36]


# In[25]:



configs_to_explore = {
'VALID_BATCH_SIZE' : {'copy_from':'TRAIN_BATCH_SIZE'}, #4,

# drop out:
"USE_MULTI_SAMPLE_DROPOUT": {'to_explore':[False, True]}, #https://github.com/oleg-yaroshevskiy/quest_qa_labeling/blob/master/step5_model3_roberta_code/model.py#L119
'DROPOUT': {
            'to_explore':[0,0.25,0.5],
            'on_false':'USE_MULTI_SAMPLE_DROPOUT'
            }, #0.5, #0, #
"USE_MULTI_SAMPLE_DROPOUT_RATE": {
            'to_explore':[0.75, 0.5],
            'on_true':'USE_MULTI_SAMPLE_DROPOUT'
            },
"USE_MULTI_SAMPLE_DROPOUT_SAMPLE":  {
            'to_explore':[4, 8],
            'on_true':'USE_MULTI_SAMPLE_DROPOUT'
            },

"MODEL_CLASS": {
            'to_explore':["CRPModel","AttentionHeadedModel", "SimpleCNNHeadedModel"]},

# pretraining
"PRETRAIN_TO_LOAD": {'to_explore':[None, f"{input_path}robertaitpt", 'crp/data//ctpt-pretrained-1622132310/model_0/']}, #

# optimizer
'LR':{'to_explore':[2e-5, 4e-5, 8e-5]},#4e-5,#4e-5, #
"USE_SIMPLE_OPTIMIZER": {'to_explore':[True, False]},

'ADD_AUGMENT':{'to_explore':[None,
                             "crp/data/train_sent_sample_0.5.csv", # training_log_1622794364.log
                             ["crp/data/train_sent_sample_0.5.csv", 'crp/data/train_sent_count_sample_1.csv', 'crp/data/train_sent_count_sample_3.csv'],
                             ["crp/data/splitted_url_aug_sents_9.csv", "crp/data/train_sent_sample_0.5.csv"]
                            ]},

# backbone groups:
"BACKBONE_GROUPS":{'is_group':True,
                  'groups_to_explore':[
                      {
                          'TOKENIZER':'roberta-base',
                          'BERT_PATH':'roberta-base'
                      },
                      {
                          'TOKENIZER':'microsoft/deberta-base',
                          'BERT_PATH':'microsoft/deberta-base'
                      }
                  ]}
}
configs_to_explore = {**base_config, **configs_to_explore}
#if TEST_MODE:
#    config['FOLDS'] = 2
#    config['EPOCHS'] = 1

assert 'SEED' not in configs_to_explore

import pprint


fields_for_hash = list([k for k,v in configs_to_explore.items() if not (isinstance(v, dict) and 'is_group' in v)])
fields_for_hash.append('SEED')

keys_to_exclude_in_hash = ['GPU_PARALLEL_IDS', 'EMBED_OTHER_GPU']

#for k in keys_to_exclude_in_hash:
fields_for_hash.remove('GPU_PARALLEL_IDS')
fields_for_hash.remove('EMBED_OTHER_GPU')

assert 'SEED' in fields_for_hash


fields_for_hash_no_seed = fields_for_hash.copy()
fields_for_hash_no_seed.remove('SEED')
assert 'SEED' not in fields_for_hash_no_seed

config_to_explore_list = generate_config_list(configs_to_explore)


# In[26]:


from random import shuffle

#assert len(config_to_explore_list) == 85, f"{len(config_to_explore_list)}"

shuffle(config_to_explore_list)


# In[86]:


# check existing:
def conf_explored(fields_for_hash, df,conf_dict,debug=False):
    explored = get_matched_rows(fields_for_hash, df,conf_dict, debug)
    to_ret= explored is None or len(explored) == 0

    return not to_ret
def get_matched_rows(fields_for_hash, df,conf_dict,debug=False):

    assert isinstance(conf_dict, dict)

    explored = None

    for f in fields_for_hash:
        #print(conf_dict[f])
        not_in_dict = f not in conf_dict
        # not in both
        if not_in_dict and f not in df.columns:
            continue
        # not in df
        elif f not in df.columns:
            assert False, f'should add {f} manually using add_new_config_item.ipynb'
        # not in dict only
        elif not_in_dict:
            tmp = df[f].isna()
        elif (not isinstance(conf_dict[f], list)) and pd.isna(conf_dict[f]):
            #print(conf_dict[f])
            tmp = df[f].isna()
        else:
            tmp = df[f].apply(lambda x: (isinstance(x, list) == isinstance(conf_dict[f], list)  and (x == conf_dict[f])))

        if explored is None:
            explored = tmp
        else:
            explored = explored & tmp
        if  debug:
            print(f"after matching {f}/{conf_dict[f]}:\n{df[explored]}\n")
    return df[explored] if explored is not None else []


# In[ ]:


def get_explored_df(filter_invalid_loss=0.43):

    pathlist = Path("crp/explored_df").glob('**/training_result.pkl.*')
    to_ret = None


    with FileLock("crp/explored_df/explored_df.lock"):
        #print("explored_df lock obtained.")
        pl = [str(path) for path in pathlist]
        #print(pl)

        merged_to_remove = []
        for path in tqdm(pl, desc="loading explored dfs", total=len(pl)):
            #print(f"fetching {path}")
            # because path is object not string
            path_in_str = str(path)

            if ".soft_remove" in path_in_str:
                continue

            current_df = pd.read_pickle(path_in_str, compression=None)

            if to_ret is None:
                to_ret = current_df
            else:
                to_ret = to_ret.append(current_df)

            if len(current_df) > 0:
                merged_to_remove.append(path_in_str)

        to_ret = to_ret.drop_duplicates(subset='saving_ts')
        if filter_invalid_loss is not None:
            to_ret = to_ret[to_ret.total_loss > filter_invalid_loss]

        if len(pl) > 1:

            save_file = store_explored_df_no_lock(to_ret)

            assert to_ret['saving_ts'].nunique() == pd.read_pickle(save_file, compression=None).shape[0]

            for path in tqdm(merged_to_remove, desc="removing merged explored dfs", total=len(pl)):
                path_in_str = str(path)
                os.rename(path_in_str, path_in_str+".soft_remove")
                #print(f"{path_in_str} deleted")
            print("merged to " + save_file)
    return to_ret
def store_explored_df(df):
    with FileLock("crp/explored_df/explored_df.lock"):
        print("explored_df lock obtained.")
        return store_explored_df_no_lock(df)

def store_explored_df_no_lock(df):
    save_file = f'crp/explored_df/training_result.pkl.{int(time.time())}.{random.randint(0,65535)}'
    df = df.drop_duplicates(subset='saving_ts')
    df.to_pickle(save_file, compression=None)
    logging.info(f"{save_file} updated.")
    print(f"{save_file} updated.")
    return save_file

def add_col_explored_df(col, value):

    pathlist = Path("crp/explored_df").glob('**/training_result.pkl.*')
    with FileLock("crp/explored_df/explored_df.lock"):
        print("explored_df lock obtained.")
        for path in pathlist:
            # because path is object not string
            path_in_str = str(path)
            if ".soft_remove" in path_in_str:
                continue
            df = pd.read_pickle(path_in_str)
            if col not in df.columns:
                df[col] = value
                df.to_pickle(path_in_str)
# In[95]:
def modify_col_explored_df(col, original_val, value):

    pathlist = Path("crp/explored_df").glob('**/training_result.pkl.*')
    with FileLock("crp/explored_df/explored_df.lock"):
        print("explored_df lock obtained.")
        for path in pathlist:
            # because path is object not string
            path_in_str = str(path)
            if ".soft_remove" in path_in_str:
                continue
            df = pd.read_pickle(path_in_str)
            if col in df.columns:
                if original_val is None:
                    df[col] = df[col].apply(lambda x: value if pd.isna(x) else x)
                else:
                    df[col] = df[col].apply(lambda x: value if x == original_val else x)
                df.to_pickle(path_in_str)

def modify_col_filter_explored_df(col, val, target_col, target_val, assert_one=True):

    pathlist = Path("crp/explored_df").glob('**/training_result.pkl.*')
    with FileLock("crp/explored_df/explored_df.lock"):
        print("explored_df lock obtained.")
        for path in pathlist:
            # because path is object not string
            path_in_str = str(path)
            if ".soft_remove" in path_in_str:
                continue
            df = pd.read_pickle(path_in_str)
            if col in df.columns:
                if assert_one:
                    assert len(df.loc[df[col] == val]) == 1
                df.loc[df[col] == val, target_col] = target_val
                df.to_pickle(path_in_str)

def compare_2_confs(conf0, conf1,
                    base_config=base_config,
                    fields_for_hash=fields_for_hash,
                    training_result_file_name = None, #'crp/training_result.csv',
                    seeds=SEED_to_explore,
                    pval_bar=0.25,
                    max_rounds=10,
                    min_len=3,
                    import_file_path=None):

    print(f"comparing: {conf0} V.S. {conf1}")
    logging.warning(f"comparing: {conf0} V.S. {conf1}")
    ori_conf0 = conf0
    ori_conf1 = conf1
    conf0 = {**base_config, **conf0}
    conf1 = {**base_config, **conf1}

    assert conf0 != conf1

    rounds = 0

    pval  =1.

    explored_df = None

    #max_rounds = 10

    min_conf_len = 0

    if not (min_conf_len < min_len or (pval >= pval_bar and min_conf_len < max_rounds)):
        return "already_concluded"

    while min_conf_len < min_len or (pval >= pval_bar and min_conf_len < max_rounds):
        print(f"\n [round:{rounds}]")
        logging.warning(f"\n [round:{rounds}]")

        if explored_df is None:
            explored_df = get_explored_df() #pd.read_csv('crp/training_result.csv')

        fields_for_hash_no_seed = fields_for_hash.copy()
        fields_for_hash_no_seed.remove('SEED')

        assert 'SEED' not in fields_for_hash_no_seed

        matched_conf0 = get_matched_rows(fields_for_hash_no_seed, explored_df, conf0, debug=False)
        matched_conf1 = get_matched_rows(fields_for_hash_no_seed, explored_df, conf1)

        #print(f"matching: {matched_conf0}, {matched_conf1}")
        print(f"matched_conf0 len:{len(matched_conf0)}, matched_conf1 len:{len(matched_conf1)}")
        logging.warning(f"matched_conf0 len:{len(matched_conf0)}, matched_conf1 len:{len(matched_conf1)}")

        if rounds > 0:
            assert len(matched_conf0) > 0 and len(matched_conf1)>0

        min_conf_len = min(len(matched_conf0),len(matched_conf1))

        if len(matched_conf0) > 0 and len(matched_conf1)>0:

            pval, _, pc = calculate_p_value_mde(matched_conf0.total_loss, matched_conf1.total_loss)

            #print(f"{ori_conf0} - {matched_conf0.total_loss} ")
            #print(f"{ori_conf1} - {matched_conf1.total_loss} ")

            print(f"{ori_conf0} - mean: {matched_conf0.total_loss.mean()} std: {matched_conf0.total_loss.std()}")
            print(f"{ori_conf1} - mean: {matched_conf1.total_loss.mean()} std: {matched_conf1.total_loss.std()}")
            print(f"pval: {pval}, pc: {pc}\n")
            logging.warning(f"{ori_conf0} - mean: {matched_conf0.total_loss.mean()} std: {matched_conf0.total_loss.std()}")
            logging.warning(f"{ori_conf1} - mean: {matched_conf1.total_loss.mean()} std: {matched_conf1.total_loss.std()}")
            logging.warning(f"pval: {pval}, pc: {pc}\n")

            if min_conf_len >= min_len and (pval < pval_bar or min_conf_len >= max_rounds):
                print(f"done")
                return "already_concluded"
                break

        next_seed_idx0 = 0
        next_seed_idx1 = 0

        while seeds[next_seed_idx0] in matched_conf0['SEED'].values:
            next_seed_idx0+=1
        while seeds[next_seed_idx1] in matched_conf1['SEED'].values:
            next_seed_idx1+=1

        print(f"found seed {seeds[next_seed_idx0]} besides {matched_conf0['SEED'].values}")

        print(f"1 more round of testing at seeds: {seeds[next_seed_idx0]} and {seeds[next_seed_idx1]}")

        logging.warning(f"found seed {seeds[next_seed_idx0]} besides {matched_conf0['SEED'].values}")

        logging.warning(f"1 more round of testing at seeds: {seeds[next_seed_idx0]} and {seeds[next_seed_idx1]}")

        if len(matched_conf0) <= len(matched_conf1):
            explored_df = run_with_record(conf0, seeds[next_seed_idx0], explored_df,
                        fields_for_hash=fields_for_hash, import_file_path=import_file_path)

        if len(matched_conf0) >= len(matched_conf1):
            explored_df = run_with_record(conf1, seeds[next_seed_idx1], explored_df,
                        fields_for_hash=fields_for_hash, import_file_path=import_file_path)

        rounds +=1


    print(f"concluded: {conf0} V.S. {conf1}")
    logging.warning(f"concluded: {conf0} V.S. {conf1}")


# In[96]:


def run_with_record(conf, seed, input_df=None,fields_for_hash=fields_for_hash, import_file_path=None):
    #training_result_file_name = 'crp/training_result.csv'

    if input_df is None:
        tmpdf = get_explored_df() #pd.read_csv(training_result_file_name)
    else:
        tmpdf = input_df

    assert 'SEED' in fields_for_hash
    if conf_explored(fields_for_hash, tmpdf, {**conf, **{'SEED':seed}}):
        print(f"already explored: seed:{seed}, and conf: {conf}")
        return tmpdf

    conf['SEED']=seed
    row = copy.deepcopy(conf)
    saving_ts, saving_dir, total_loss, all_folds_loss = run(conf, import_file_path=import_file_path)
    row['saving_ts']=saving_ts
    row['saving_dir']=saving_dir
    row['total_loss']=total_loss
    row['all_folds_loss']=all_folds_loss
    row['ts']=int(time.time())
    row['config_id'] = 0


    # open again to avoid overwriting
    if input_df is None:
        tmpdf = get_explored_df() #pd.read_csv(training_result_file_name)
    else:
        tmpdf = input_df

    existing_tss = tmpdf.saving_ts.values
    print(f"existing id count: {len(existing_tss)}")
    # make sure the same columns:
    tmpdf = tmpdf.append([row])
    new_rows = tmpdf[~tmpdf.saving_ts.isin(existing_tss)]
    assert(len(new_rows) == 1)
    assert(new_rows['saving_ts'].values[0] == saving_ts)
    print(f"new row count: {len(new_rows)}")
    #print(tmpdf)
    #print('saving...')

    store_explored_df(new_rows)

    return tmpdf

def run_with_record_assign_seed(conf, input_df=None,fields_for_hash=fields_for_hash, import_file_path=None, seeds=SEED_to_explore):
    explored_df = get_explored_df() #pd.read_csv('crp/training_result.csv')

    fields_for_hash_no_seed = fields_for_hash.copy()
    fields_for_hash_no_seed.remove('SEED')

    assert 'SEED' not in fields_for_hash_no_seed

    matched_conf0 = get_matched_rows(fields_for_hash_no_seed, explored_df, conf, debug=False)

    next_seed_idx0 = 0

    while seeds[next_seed_idx0] in matched_conf0['SEED'].values:
            next_seed_idx0+=1


    print(f"found seed {seeds[next_seed_idx0]} besides {matched_conf0['SEED'].values}")

    return run_with_record(conf, seeds[next_seed_idx0], input_df,fields_for_hash, import_file_path)

# In[27]:

def multi_run_with_record_assign_seed(conf, count=10, import_file_path=None):

    for i in range(count):
        print(f"exploring conf: {i}/{count}...\n\n")
        logging.info(f"exploring conf: {i}/{count}...\n\n")
        run_with_record_assign_seed(conf, import_file_path=import_file_path)
        print(f"finished conf: {i}/{count}...\n\n")
        logging.info(f"finished conf: {i}/{count}...\n\n")




#if TRAIN_MODE:


def randome_explore(seeds_count=10):
    #training_result_file_name = 'crp/training_result.csv'
    #training_result_file_name_rename = 'crp/training_result.csv.old'

    explored_df = None

    if os.path.exists(training_result_file_name):

        explored_df = get_explored_df() #pd.read_csv('crp/training_result.csv')


        #get_ipython().system('mv {training_result_file_name} {training_result_file_name_rename}')



    for ix, conf in enumerate(config_to_explore_list):

        for seed in SEED_to_explore[:seeds_count]:

            if explored_df is not None and conf_explored(fields_for_hash, explored_df, conf):
                print(f"{conf} already explored, skip.")
                continue
            explored_pairs.append((seed, ix))

            logging.info(f"\n{red}-------training {ix}/{len(config_to_explore_list)} @ seed:{seed}-------{blk}\n")
            print(f"\n{red}-------training {ix}/{len(config_to_explore_list)} @ seed:{seed}-------{blk}\n")

            run_with_record(conf, seed)


# In[28]:





# ## gen .sh script and .json dataset metadata file

# In[ ]:


import json

def gen_scripts(SAVING_TS = 1622161186, TS_IN_TITLE=False):
    if SAVING_TS is not None:
        row = get_explored_df() #pd.read_csv('crp/training_result.csv')
        row = row[row.saving_ts==SAVING_TS]
        saving_dir = row.saving_dir.values[0]
        saving_ts = row.saving_ts.values[0]

        metadata = {
          "title": f"single_pretrained_model_{SAVING_TS}" if TS_IN_TITLE else "pretrained_model_1621892031",
          "id": f"ziyou1guangchang/pretrained-model-{SAVING_TS}" if TS_IN_TITLE else f"ziyou1guangchang/pretrained-model-1621892031",
          "licenses": [
            {
              "name": "CC0-1.0"
            }
          ]
        }
        with open(f"{saving_dir}/dataset-metadata.json",'w') as f:
            json.dump(metadata, f)

        hostname = get_ipython().getoutput('hostname')
        hostname = hostname[0]

        dirname = saving_dir.split('/')[-1]

        print(f"run the following commend on mac:\n scp -r gavin_li@{hostname}.inst.aws.airbnb.com:~/redspot_home/crp/data/{dirname} ./")
        print(f"\n\nrun the following commend: \nkaggle datasets version -m {dirname} -p ./{dirname} --dir-mode zip")
        print(f"\n\npaste the ts: {saving_ts} to the assertion code below")


# In[ ]:





# # pred..

# In[ ]:





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
    print(pretrain_paths)
    pred_sum = np.zeros((len(df)))

    print(f'loadding tokenizer from {pretrain_paths[0]}')

    # fix GPT2 save pretrain issue...
    print(f"loading tokenizer for {config['TOKENIZER']}")
    if config['TOKENIZER'] == 'microsoft/deberta-base':
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
            if config['TOKENIZER'] in ["deepset/roberta-base-squad2","chkla/roberta-argument","roberta-base", "deepset/roberta-large-squad2"]:
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
            batch_size = config['VALID_BATCH_SIZE'],
            num_workers = 2,
            drop_last=False,
        )


    scores = []
    embeds = []

    for p in pretrain_paths:

        #model = AutoModelForSequenceClassification.from_pretrained(p,num_labels=1)
        model_class = getattr(model_import, config['MODEL_CLASS'])

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

def diff_dict(dict1, dict2, to_exclude = keys_to_exclude_in_hash):
    dict1 = copy.copy(dict1)
    dict2 = copy.copy(dict2)
    for k in to_exclude:
        if k in dict1:
            del dict1[k]
        if k in dict2:
            del dict2[k]
    try:
        set1 = set([(k,tuple(v)) if isinstance(v, list) else (k,v) for k,v in dict1.items()])
    except TypeError as te:
        print(te)
        raise te
    try:
        set2 = set([(k,tuple(v)) if isinstance(v, list) else (k,v) for k,v in dict2.items()])
    except TypeError as te:
        print(te)
        raise te
    return {k:dict1[k] for k in dict(set1 ^ set2).keys()}


def conf_fill_na(conf_dict):
    for k,v in conf_dict.items():
        if pd.isna(conf_dict[k]):
            conf_dict[k] = None
    return conf_dict


def get_conf_name_by_diff_base(x):
    xd = copy.copy(x.to_dict())
    keys = list(xd.keys())
    for k in keys:
        if k not in fields_for_hash:
            del xd[k]
        else:
            # fill na:
            if pd.isna(xd[k]):
                xd[k] = None
    return diff_dict(xd, base_config)


def convert_list_to_tuple(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    return df

def get_conf_rank(precondition=None, max_seed_count=None):

    explored_df = get_explored_df() #pd.read_csv('crp/training_result.csv')
    #explored_df = explored_df.fillna(None)
    explored_df = convert_list_to_tuple(explored_df)

    if precondition is not None:
        for k,v in precondition.items():
            if v is None:
                explored_df = explored_df[explored_df[k].isna()]
            else:
                explored_df = explored_df[explored_df[k] == v]

    if max_seed_count is not None:
        seeds_list = SEED_to_explore[:max_seed_count]

        explored_df = explored_df[explored_df.SEED.isin(seeds_list)]

    fields_for_hash_no_seed = fields_for_hash.copy()
    fields_for_hash_no_seed.remove('SEED')


    # dedup seeds...
    explored_df = explored_df.sort_values('saving_ts').copy()
    print(f"before dedup seed:{len(explored_df)}")
    explored_df = explored_df.drop_duplicates(['SEED'] + fields_for_hash_no_seed, keep='last')
    print(f"after dedup seed:{len(explored_df)}")



    sorted_df = explored_df.groupby(fields_for_hash_no_seed, as_index=False, dropna=False).agg(mean_loss=('total_loss',"mean"),
                                                                                               seeds=('SEED', lambda x: x.to_list()),
                                                                                               tss=('saving_ts', lambda x: x.to_list()),
                                                                                               mean_all_folds=('all_folds_loss', 'mean'),
                                                                                               run_ts=('saving_ts','min'),
                                                                                 run_count=('total_loss','count'),
                                                                                 loss_std=('total_loss','std')).sort_values('mean_loss')

    if len(sorted_df) == 0:
        return sorted_df
    #print(sorted_df.apply(get_conf_name_by_diff_base, axis=1))
    sorted_df['conf_name'] = sorted_df.apply(get_conf_name_by_diff_base, axis=1)
    return sorted_df




def explore_confs(precondition,configs_to_explore, max_test_count = 40):

    preconditioned_configs_to_explore = {**base_config, **precondition, **configs_to_explore}
    conf_list = generate_config_list(preconditioned_configs_to_explore)
    conf_list = constraint_conflist(conf_list, precondition)

    test_count = 0
    all_concluded = False
    last_best_conf = None

    while not all_concluded and test_count < max_test_count:
        test_count += 1

        explored_confs = get_conf_rank(precondition)[fields_for_hash_no_seed].to_dict('records')

        if len(explored_confs) == 0:
            best_conf = conf_list[0]
        else:
            #print(explored_confs)
            best_conf = explored_confs[0]
            best_conf = conf_fill_na(best_conf)

        if best_conf != last_best_conf:
            print(f"a new best config: {best_conf}")
            logging.info(f"a new best config: {best_conf}")
            last_best_conf = best_conf

        # sort by diff small to big
        sorted(conf_list, key=lambda x: len(diff_dict(x, best_conf).keys()))

        all_concluded = True
        for c in conf_list:
            if {**base_config, **c} == {**base_config, **best_conf}:
                continue

            print(f"\n[{test_count}]: {grn}smallest diff config to compare: {diff_dict(c, best_conf)}{blk}\n")
            logging.info(f"\n{grn}smallest diff config to compare: {diff_dict(c, best_conf)}{blk}\n")
            ret = compare_2_confs(c, best_conf, import_file_path=[f"crp/{import_file_name}.py",
                                                             f"crp/{import_file_name}.py"])
            if ret != "already_concluded":
                all_concluded = False
                break


