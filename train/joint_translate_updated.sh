#!/bin/bash
echo `date`
infname=$1
outfname=$2
src_lang=$3
tgt_lang=$4
exp_dir=$5
ref_fname=$6

SRC_PREFIX='SRC'
TGT_PREFIX='TGT'

#`dirname $0`/env.sh
SUBWORD_NMT_DIR='subword-nmt'
model_dir=$exp_dir/model
data_bin_dir=$exp_dir/final_bin

### normalization and script conversion

echo "Applying normalization and script conversion"
input_size=`python scripts/preprocess_translate.py $infname $outfname.norm $src_lang true`
echo "Number of sentences in input: $input_size"

### apply BPE to input file

echo "Applying BPE"
python $SUBWORD_NMT_DIR/subword_nmt/apply_bpe.py \
    -c $exp_dir/vocab/bpe_codes.32k.${SRC_PREFIX} \
    --vocabulary $exp_dir/vocab/vocab.$SRC_PREFIX \
    --vocabulary-threshold 5 \
    < $outfname.norm \
    > $outfname._bpe

# not needed for joint training
# echo "Adding language tags"
python scripts/add_tags_translate.py $outfname._bpe $outfname.bpe $src_lang $tgt_lang

### run decoder

echo "Decoding"

src_input_bpe_fname=$outfname.bpe
tgt_output_fname=$outfname

fairseq-interactive  $data_bin_dir \
    -s $SRC_PREFIX -t $TGT_PREFIX \
    --distributed-world-size 1 --fp16 \
    --max-tokens 10240 \
    --path $model_dir/checkpoint_best.pt \
    --batch-size 32 --buffer-size 1024 --data-buffer-size 8 --beam 5 --remove-bpe \
    --skip-invalid-size-inputs-valid-test \
    --user-dir model_configs \
    --input $src_input_bpe_fname  >  $tgt_output_fname.log 2>&1


echo "Extracting translations, script conversion and detokenization"
# this part reverses the transliteration from devnagiri script to target lang and then detokenizes it.
python scripts/postprocess_translate.py $tgt_output_fname.log $tgt_output_fname $input_size $tgt_lang true
echo "Translation completed"