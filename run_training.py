
from train import train
from util.config import TrainingConfig
from train.train import run
new_config = {
    'USE_TRAIN_AS_TEST': True,
    'STRATEFIED': False,
    'ADD_AUGMENT': None,
    'TOKENIZE_ALL': False,
    'EVAL_PERIOD': 100,
    'TRAIN_EVAL_PERIOD': 100,

}

config = TrainingConfig(new_config)

run(config)