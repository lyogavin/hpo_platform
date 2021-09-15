import pprint
import json

default_runtime_config = {
    'DEBUG_PRINT':False,
    'EVAL_PERIOD':100,
    'TRAIN_EVAL_PERIOD':100,
    'SAVING_THRESHOLD':0.9,
    'EMBED_OTHER_GPU':False,
    'GPU_PARALLEL_IDS':None,
    'VALID_BATCH_SIZE':16,
    'DATA_ROOT_PATH':'../chaii/input/',
    'OUTPUT_ROOT_PATH':'../chaii/output/',
    'PRED_ROOT_PATH':'../chaii/output/',
    'FORCE_NUM_LOADER_WORKER':None
}
default_train_config = {
    # test run...
    'TEST_RUN':False,

    # optimizer
    'LR':1e-5,
    'LAYERED_OPT_ENABLED':'',
    'LAYERED_OPT_DEFAULT_WEIGHT_DECAY':None,
    'LAYERED_OPT_DEFAULT_LR':None,
    'OPTIMIZER_NAME':'AdamW',
    'WEIGHT_DECAY':0,
    'ADAMW_BETAS':None,
    'ADAMW_EPS':None,
    'DECAY_NAME':'linear',
    'WARMUP_STEPS_RATIO':0.0,
    'NUM_CYCLES':None,
    'POLY_DECAY_LR_END':1.0,
    'FREEZE_EMBED':False,

    #data
    'STRATEFIED':True,
    'USE_TRAIN_AS_TEST':False,
    'ADD_AUGMENT':None,
    'AUGMENT_SKIP_TRAINING':False,
    'AUGMENT_REWEIGHTING':False,
    'STRATEFIED_SAMPLER':False,
    'MAX_LEN':256,
    'TOKENIZE_ALL':False,
    'REMOVE_NEWLINE':False,
    'TOKENIZER_PADDING':None,
    'STRIDE':0,
    'USE_QUOREF':False,

    #data-similarity sampling
    'SIMILARIY_EMBED_MODEL':"paraphrase-multilingual-mpnet-base-v2",  # https://www.sbert.net/docs/pretrained_models.html#multi-lingual-models
    'SIM_SAMPLE_DATASETS':{'MLQA', "XQUAD","QUOREF","NEWSQA"},
    'SIM_SAMPLE_RATIO':1.0,
    'SIM_SAMPLE_BY_LANG':True,


    #model
    'TOKENIZER':None,
    'MODEL_CONFIG':None,
    'MODEL_CLASS':None,
    'PRETRAIN_TO_LOAD':None,
    'HEAD_DROPOUT':None,
    'BERT_PATH':None,


    #train
    'GRAD_ACCU_STEPS':1,
    'EPOCHS':3,
    'FOLDS':3,
    'SEED':42,
    'TRAIN_BATCH_SIZE':16,
    'SHUFFLE_TRAIN':True,
    'VALID_BATCH_SIZE':16,
    'LOSS_TYPE':None,
    'AUTO_SCALER':False,
    'RESEED_EVERY_FOLD':True,
    'STOP_AT_FOLD':-1,

}

class TrainingConfig:
    def __init__(self, config):
        self.new_config = config
        self.config = {**default_train_config, **default_runtime_config, **config}

    def __getitem__(self, key):
        assert key in default_train_config or key in default_runtime_config, f"must define default value for key: {key}"

        if key not in self.config:
            # this happens when config is recovered from a previous persisted record, use default value:
            return default_train_config[key] if key in default_train_config else default_runtime_config[key]
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value

    def __contains__(self, key):
        return key in self.config


    def __str__(self):
        return pprint.pformat(self.config)


