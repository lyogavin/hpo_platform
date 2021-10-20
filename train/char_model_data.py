

import sys
sys.path.append('../')
sys.path.append('./')

from utils.utils import *
from train.data import *

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def char_model_prepare_train_features(config, example, tokenizer):


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


class CharDataset(Dataset):
    def __init__(self, df, X, n_models=1, max_len=1500, train=True):
        self.max_len = max_len
        self.df = df.copy()
        #start_probas, end_probas

        if 'start_position' in self.df.columns:
            max_len = self.df['end_position'].max() + 1

            logging.info(f"filtering based on maxlen: {max_len}, before filter: {len(self.df)}")

            self.df = self.df[self.df.start_position < max_len]
            logging.info(f"filtering based on maxlen: {max_len}, after filter: {len(self.df)}")

            self.df = self.df.reset_index(drop=True)

            self.df['end_position'] = self.df['end_position'].apply(lambda x: min(x, max_len-1))
        else:
            max_len = self.df['context'].apply(lambda x: len(x)).max()

            logging.info(f"using max len context as maxlen: {max_len}")

        self.original_lens = [len(x) for x in X]

        self.X = pad_sequences(X, maxlen=max_len, padding='post', truncating='post')

        self.start_probas = np.zeros((len(self.df), max_len, n_models), dtype=float)
        self.end_probas = np.zeros((len(self.df), max_len, n_models), dtype=float)
        for i, p in self.df.iterrows():
            for ft in p['mapping_to_logits']:
                mapping, starts, ends = ft
                for map, start, end in zip(mapping, starts, ends):
                    if map is not None and map[0] < max_len:
                        self.start_probas[i, map[0]:min(map[1]+1, max_len)] = start
                        self.end_probas[i, map[0]:min(map[1]+1, max_len)] = end

        self.texts = self.df['context'].values
        self.ids = self.df['id'].values
        self.questions = self.df['question']



    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        to_ret = {}
        to_ret = {
            'input_ids': torch.tensor(self.X[idx], dtype=torch.long),
            #'attention_mask': torch.tensor(feature['attention_mask'], dtype=torch.long),
            #'offset_mapping': torch.tensor(feature['offset_mapping'], dtype=torch.long),
            #'sequence_ids': torch.tensor(feature['sequence_ids'], dtype=torch.long),
            'id': self.df.iloc[idx]['id'],
            'context': self.df.iloc[idx]['context'],
            'question': self.df.iloc[idx]['question'],
            # 'features_index':item
            'start_probas': torch.tensor(self.start_probas[idx]).float(),
            'end_probas': torch.tensor(self.end_probas[idx]).float(),
            'original_len': self.original_lens[idx]
        }
        if 'start_position' in self.df.columns:
            to_ret['start_position'] = torch.tensor(self.df.iloc[idx]['start_position'], dtype=torch.long)
            to_ret['end_position'] = torch.tensor(self.df.iloc[idx]['end_position'], dtype=torch.long)
            to_ret['answer_text'] = self.df.iloc[idx]['answer_text']

        return to_ret

def char_model_make_loader(
        config,
        data, split_output,
        tokenizer, fold
):
    tokenizer = Tokenizer(num_words=None, char_level=True, oov_token='UNK', lower=True)

    to_fit = pd.read_csv(f'{config["DATA_ROOT_PATH"]}/chaii-hindi-and-tamil-question-answering/train.csv')

    tokenizer.fit_on_texts(to_fit.apply(lambda x: ' '.join([x['context'], x['question'], x['answer_text']]), axis=1).values)

    len_voc = len(tokenizer.word_index) + 1
    data["start_position"] = data["answer_start"]
    data["end_position"] = data.apply(lambda x: x['start_position'] + len(x["answer_text"]), axis=1)

    train_set, valid_set = data.loc[split_output[fold][0]], data.loc[split_output[fold][1]]

    X_train = tokenizer.texts_to_sequences(data['context'].values)
    X_test = tokenizer.texts_to_sequences(data['context'].values)

    train_dataset = CharDataset(data, X_train)
    valid_dataset = CharDataset(data, X_test)
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
    return train_dataloader, valid_dataloader, train_set, valid_set, len_voc


def collate_fn(batch):
    """
    batch = [dataset[i] for i in N]
    """

    max_len = max([x['original_len'] for x in batch])

    to_ret = []
    for x in batch:
        x['input_ids'] = x['input_ids'][:max_len]
        x['start_probas'] = x['start_probas'][:max_len]
        x['end_probas'] = x['end_probas'][:max_len]

    return batch

def char_model_make_test_loader(
        config,
        tokenizer,
        df=None, data_input_path=None):

    input_path = config['DATA_ROOT_PATH'] if data_input_path is None else data_input_path

    if df is None:
        test = pd.read_csv(f'{input_path}/chaii-hindi-and-tamil-question-answering/test.csv')
    else:
        test = df


    tokenizer = Tokenizer(num_words=None, char_level=True, oov_token='UNK', lower=True)

    to_fit = pd.read_csv(f'{input_path}/chaii-hindi-and-tamil-question-answering/train.csv')

    tokenizer.fit_on_texts(to_fit.apply(lambda x: ' '.join([x['context'], x['question']]), axis=1).values)

    len_voc = len(tokenizer.word_index) + 1
    X_test = tokenizer.texts_to_sequences(test['context'].values)

    valid_sampler = SequentialSampler(test)
    valid_dataset = CharDataset(test, X_test)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config['VALID_BATCH_SIZE'],
        sampler=valid_sampler,
        num_workers=optimal_num_of_loader_workers(config),
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )
    test['input_ids'] = X_test
    return valid_dataloader, test.to_dict('records'), len_voc
