from train.metric import  *
import os
# test...
if __name__ == "__main__":
    # test post-process and jaccard...

    level = logging.INFO
    logger = logging.getLogger()
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)

    os.environ["DEBUSSY"] = "1"
    meter = AccumulateMeter()
    from utils.config import TrainingConfig
    config = TrainingConfig({'USE_TRAIN_AS_TEST': False, 'TEST_RUN':True})
    from train.data import get_data_kfold_split, make_loader
    train, split_output = get_data_kfold_split(config)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('deepset/xlm-roberta-large-squad2')

    train_loader, valid_loader, train_features, valid_features = make_loader(config, train, split_output, tokenizer, 0)
    #print(f"first 5 of train features: {train_features[:5]}")
    for idx, d in enumerate(train_loader):
        meter.update(d,
                     torch.nn.functional.one_hot(d['start_position'], num_classes=config['MAX_LEN']).float(),
                     torch.nn.functional.one_hot(d['end_position'], num_classes=config['MAX_LEN']).float(),
                     d['start_position'], d['end_position'])
        #break
    print(meter.get_metrics(tokenizer))