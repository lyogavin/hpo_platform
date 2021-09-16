#!/bin/bash

# setting up the env... (will take some time, please be patient, require active internet connection)

# clone the repo for running evaluation
git clone https://github.com/AI4Bharat/indicTrans.git
cd indicTrans
# clone requirements repositories
git clone https://github.com/anoopkunchukuttan/indic_nlp_library.git
git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git
git clone https://github.com/rsennrich/subword-nmt.git
cd ..


# Install the necessary libraries
pip install sacremoses pandas mock sacrebleu tensorboardX pyarrow indic-nlp-library
pip install mosestokenizer subword-nmt


# Install fairseq from source
git clone https://github.com/pytorch/fairseq.git

cd fairseq
# !git checkout da9eaba12d82b9bfc1442f0e2c6fc1b895f4d35d
pip install --editable ./
python setup.py build_ext --inplace
cd ..

# this import might not work without restarting runtime
# from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils

# english-to-indic tranlated (machine generated dataset)
# model from https://indicnlp.ai4bharat.org/indic-trans/
# will take some time, so sit back and sip your coffee! (please avoid downloading again and again, push it to a kaggle dataset :pray:)

wget https://storage.googleapis.com/samanantar-public/V0.2/models/en-indic.zip
unzip ./en-indic.zip
#!rm -f en-indic.zip

pip install transformers sentencepiece