
from train import train
from utils.config import TrainingConfig
from train.train import run
new_config = {
    'USE_TRAIN_AS_TEST': True,
    'STRATEFIED': False,
    'ADD_AUGMENT': None,
    'TOKENIZE_ALL': False,
    'EVAL_PERIOD': 100,
    'TRAIN_EVAL_PERIOD': 100,
    'TEST_RUN': True,
    'TOKENIZER': 'deepset/xlm-roberta-large-squad2',
    'BERT_PATH': "deepset/xlm-roberta-large-squad2",
    'MODEL_CONFIG': "deepset/xlm-roberta-large-squad2",
    'MODEL_CLASS': 'SpanningQAModel',
}

config = TrainingConfig(new_config)

run(config)