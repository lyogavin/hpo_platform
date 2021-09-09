from utils import utils
from train import train
from utils.config import TrainingConfig
from train.infer import infer_and_gen_submission, download_saving
import sys

utils.logging.info("started infer...")

if __name__ == "__main__":

    assert len(sys.argv) > 1, 'need to specify saving ts to infer or ts and download url to download'

    saving_ts=sys.argv[1]

    test_on_training = False
    for arg in sys.argv:
        if "--TEST_ON_TRAINING" == arg:
            test_on_training = True

    if not test_on_training and len(sys.argv) > 2:
        url = sys.argv[2]
        download_saving(url, saving_ts)
    else:

        infer_and_gen_submission(saving_ts, TRAIN_MODE=False, TEST_ON_TRAINING=test_on_training, gen_file=True)