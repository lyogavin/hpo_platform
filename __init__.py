import sys
sys.path.append('./')
from utils.utils import logging

from train.infer import gen_submission, download_saving


def infer_df(df, pretarined_path, nbest):
    return gen_submission(pretarined_path, None, df, False, False, False, nbest)