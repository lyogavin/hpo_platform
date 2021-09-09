
from train import train
from utils.config import TrainingConfig
from train.infer import infer_and_gen_submission, download_saving
import sys

if __name__ == "__main__":

    assert len(sys.argv) > 1, 'need to specify saving ts to infer or ts and download url to download'

    saving_ts=sys.argv[1]

    if len(sys.argv) > 2:
        url = sys.argv[2]
        download_saving(url, saving_ts)
    else:
        infer_and_gen_submission(saving_ts, TRAIN_MODE=False, TEST_ON_TRAINING=False, gen_file=True)