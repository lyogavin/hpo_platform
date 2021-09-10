import pickle
from pathlib import Path
from exp_record_store import *
from utils.config import default_train_config
import pandas as pd

def get_exp_dict(rec):
  to_ret = {}

  # train config keys:
  for key in default_train_config.keys():
    to_ret[key] = rec.config[key]
  # perf keys:
  to_ret['saving_ts'] = rec.saving_ts
  to_ret['saving_dir'] = rec.saving_dir
  to_ret['git_head_id'] = rec.git_head_id
  to_ret['mean_jaccard'] = rec.get_mean_jaccard()
  to_ret['mean_loss'] = rec.get_mean_loss()
  return to_ret

def get_df_from_exp_recs(root_path):

  pathlist = Path(root_path).glob('**/*.pickle')

  dicts = []
  for path in pathlist:
      # because path is object not string
      path_in_str = str(path)
      #print(path_in_str)
      with open(path_in_str, 'rb') as handle:
          exp = pickle.load(handle)
          #print(exp)
          dicts.append(get_exp_dict(exp))

  dat = pd.DataFrame.from_records(dicts)

  return dat


def present_recs(df, filters=None):
  if filters is not None:
    for k,v in filters.items():
      df = df[df[key] == v]

  common_cols = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
  common_dict = {col: df[col].values[0] for col in common_cols}
  other_cols = [col for col in df.columns if col not in common_cols]

  return df[other_cols], common_dict