import numpy as np
import os
import pickle
import sys
sys.path.append('../')
from utils.utils import logging
class ExpRecord:
    def __init__(self):
        self.fold_best_jaccards = {}
        self.fold_best_train_losses = {}

    def set_info(self,
                 saving_ts,
                 saving_dir,
                 logging_file_path,
                 git_head_id,
                 config
                 ):
        self.saving_ts = saving_ts
        self.saving_dir = saving_dir
        self.logging_file_path = logging_file_path
        self.git_head_id = git_head_id
        self.config = config

    def update_fold(self, fold, best_jaccard, best_train_loss):
        self.fold_best_jaccards[fold] = best_jaccard
        self.fold_best_train_losses[fold] = best_train_loss

    def get_mean_jaccard(self):
        if len(self.fold_best_jaccards.values()) == 0:
            return 0.
        mean_jaccard = np.array(list(self.fold_best_jaccards.values())).mean()
        return mean_jaccard

    def get_mean_loss(self):
        if len(self.fold_best_train_losses.values()) == 0:
            return 0.
        #print(self.fold_best_train_losses)
        mean_loss = np.array(list(self.fold_best_train_losses.values())).mean()
        return mean_loss

    def __str__(self):
        return f"saving_ts: {self.saving_ts}, saving_dir: {self.saving_dir}, logging_file_path: {self.logging_file_path} " \
            f"git_head_id: {self.git_head_id} mean_jaccard: {self.get_mean_jaccard():.4f} mean_loss: {self.get_mean_loss():.4f}\n" \
            f"jaccards for folds: {self.fold_best_jaccards}\n" \
            f"losses for folds: {self.fold_best_train_losses}"
    def __repr__(self):
        return self.__str__()

    def persist(self):
        #ensure dir...
        os.makedirs(self.config['OUTPUT_ROOT_PATH'] + 'exp_records/', exist_ok=True)

        pickle_file = self.config['OUTPUT_ROOT_PATH'] + 'exp_records/' + f'exp_rec_{self.saving_ts}.pickle'
        if self.config['TEST_RUN']:
            pickle_file = pickle_file + '.test'
        with open(pickle_file, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info(f"exp record persisted as {pickle_file}")
        return pickle_file


# test...
if __name__ == "__main__":
    logging.info(f"test....")
    import time
    from utils.config import TrainingConfig
    rec = ExpRecord()
    rec.set_info(int(time.time()), "/tmp/test_save", '', 'afjdklsa', TrainingConfig({}))
    rec.update_fold(0, 0.3, 0.2)
    rec.update_fold(1, 0.1, 0.2)
    rec.update_fold(2, 0.2, 0.2)

    logging.info(f"to persist: {rec}")
    file = rec.persist()
    with open(file, 'rb') as handle:
        loaded = pickle.load(handle)
    logging.info(f"from persisted: {loaded}")
