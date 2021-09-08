
from train import train
from utils.config import TrainingConfig
from train.infer import infer_and_gen_submission
import sys

if __name__ == "__main__":

    assert len(sys.argv) > 1, 'need to specify saving ts'

    saving_ts=sys.argv[1]
    infer_and_gen_submission(saving_ts, TRAIN_MODE=False, TEST_ON_TRAINING=True, gen_file=True)