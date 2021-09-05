import numpy as np

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
        mean_jaccard = np.array(self.fold_best_jaccards).mean()
        return mean_jaccard

    def get_mean_loss(self):
        mean_loss = np.array(self.fold_best_train_losses).mean()
        return mean_loss

    def __str__(self):
        return f"saving_ts: {self.saving_ts}, saving_dir: {self.saving_dir}, logging_file_path: {self.logging_file_path} " \
            f"git_head_id: {self.git_head_id} mean_jaccard: {self.get_mean_jaccard()} mean_loss: {self.get_mean_loss()}"