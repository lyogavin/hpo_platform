# install: pip install -U sentence-transformers
import pandas as pd
from sentence_transformers import SentenceTransformer, util

import sys
sys.path.append('../')
sys.path.append('./')

from utils import utils
from data import *
from utils.config import *
logging = utils.logging
import math
import numpy as np



def sample_top_n_for_row(model, context, question, from_embeds, n, lang):
    embed_context = model.encode(context)
    embed_question = model.encode(question)

    from_embeds = from_embeds[from_embeds.language == lang]

    if len(from_embeds) == 0:
        return []

    distances = from_embeds.apply(lambda x: util.cos_sim(embed_context, x['context_embed']).item() +
                                                            util.cos_sim(embed_question, x['question_embed']).item(), axis=1)

    #print(f"setting similarity for {embed_context} {embed_question}... to {distances}")

    #from_embeds['similarity'] = distances

    top_n_idx = np.argsort(distances)[-n:]
    return from_embeds.iloc[top_n_idx].index.to_list()


def add_embed_cols(df_from, model):
    df_from['context_embed']  = df_from['context'].apply(lambda x: model.encode(x))
    df_from['question_embed']  = df_from['question'].apply(lambda x: model.encode(x))

def get_similarity_sample(df_base, config, from_sample=None):
    df_from = get_datasets(config, config['SIM_SAMPLE_DATASETS'])
    if from_sample is not None:
        df_from = df_from.sample(frac=from_sample)

    model = SentenceTransformer(config['SIMILARIY_EMBED_MODEL'])
    add_embed_cols(df_from, model)

    langs = ['hindi', 'tamil']

    assert df_base['language'].apply(lambda x: x in langs).all()
    assert df_from['language'].apply(lambda x: x in langs).all()

    #assert df_base['language'].nunique() ==2
    #assert df_from['language'].nunique() ==2

    df_hindi_base = df_base[df_base['language'] == 'hindi']
    df_tamil_base = df_base[df_base['language'] == 'tamil']

    total_sample_count = len(df_base) * config['SIM_SAMPLE_RATIO']
    total_sample_count_hindi = total_sample_count * (len(df_hindi_base)/ (len(df_hindi_base) + len(df_tamil_base)))
    total_sample_count_tamil = total_sample_count * (len(df_tamil_base)/ (len(df_hindi_base) + len(df_tamil_base)))

    # sample on round up for each row first
    per_row_sample_count_hindi_rounded = math.ceil(total_sample_count_hindi/len(df_base))
    per_row_sample_count_tamil_rounded = math.ceil(total_sample_count_tamil/len(df_base))

    per_row_sample_count_lang_rounded = {'hindi':per_row_sample_count_hindi_rounded,
                                         'tamil': per_row_sample_count_tamil_rounded}

    sampled_dicts = []
    for lang in langs:
        for i,row in df_base.iterrows():
            sampled_rows = sample_top_n_for_row(model, row['context'], row['question'], df_from,
                                                per_row_sample_count_lang_rounded[lang],
                                                lang)
            sampled_dicts.extend(sampled_rows)

    to_ret = df_from.loc[sampled_dicts]

    # re-down-sample to bring down to wanted size
    to_ret_hindi = to_ret[to_ret['language'] == 'hindi']
    if len(to_ret_hindi) > 0:
        to_ret_hindi = to_ret_hindi.sample(frac=total_sample_count_hindi/len(to_ret_hindi))
    to_ret_tamil = to_ret[to_ret['language'] == 'tamil']
    if len(to_ret_tamil) > 0:
        to_ret_tamil = to_ret_tamil.sample(frac=total_sample_count_tamil/len(to_ret_tamil))

    to_ret = to_ret_hindi.append(to_ret_tamil)
    return to_ret







if __name__ == "__main__":

    TEST_COVERAGE_ONLY = True
    TEST=False

    config = TrainingConfig({'SIM_SAMPLE_DATASETS': {'MLQA', "XQUAD"},
                             'SIMILARIY_EMBED_MODEL': 'multi-qa-MiniLM-L6-cos-v1',
                             'SIM_SAMPLE_RATIO':2.0})


    if not TEST_COVERAGE_ONLY:
        model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

        #Sentences are encoded by calling model.encode()
        emb1 = model.encode("This is a red cat with a hat.")
        emb2 = model.encode("Have you seen my red cat?")

        cos_sim = util.cos_sim(emb1, emb2)
        print("Cosine-Similarity:", cos_sim)


        model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

        query_embedding = model.encode('How big is London')
        passage_embedding = model.encode(['London has 9,787,426 inhabitants at the 2011 census',
                                          'London is known for its finacial district'])

        print("Similarity:", util.dot_score(query_embedding, passage_embedding))


        # test top n similarity...
        df_from = get_datasets(config, config['SIM_SAMPLE_DATASETS']).sample(n=50)
        model = SentenceTransformer(config['SIMILARIY_EMBED_MODEL'])
        add_embed_cols(df_from, model)

        for i in range(1):
            ids = sample_top_n_for_row(df_from.iloc[i]['context'], df_from.iloc[i]['question'],
                                       df_from, 5, 'hindi')
            df_from['similarity'] = df_from.apply(
                lambda x: util.cos_sim(model.encode(x['context']), model.encode(df_from.iloc[i]['context'])).item() +
                          util.cos_sim(model.encode(x['question']), model.encode(df_from.iloc[i]['question'])).item(),
                axis=1)
            #print(df_from.loc[ids[-1]]['similarity'])
            #print(df_from[df_from['similarity'] > df_from.loc[ids[-1]]['similarity']])
            assert (df_from['similarity'] > df_from.loc[ids[-1]]['similarity']).sum() < 5

        # test distribution...
        if TEST:
            df = get_similarity_sample(df_from.sample(frac=0.5), config)
        else:
            df = get_similarity_sample(df_from, config)

        assert np.isclose(len(df), len(df_from), 1e-3)
        assert np.isclose(len(df[df.language == 'hindi']), len(df_from[df_from.language == 'hindi']), 1e-3)

    print(f"test sampling coverage...")
    # test sampling coverage:
    df_from = get_datasets(config, config['SIM_SAMPLE_DATASETS'])

    train, _ = get_train_and_test_df(root_path='../chaii/input/')

    if TEST:
        sample_df = get_similarity_sample(train.sample(frac=0.05), config, from_sample=0.05)
    else:
        sample_df = get_similarity_sample(train, config)

    print(f"default sampling covers: {df_from.index.isin(sample_df.index.values).mean()}")









