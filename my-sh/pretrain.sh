#!/bin/bash
#PBS -j oe
#PBS -q GPU-L
# https://www.jaist.ac.jp/iscenter/mpc/kagayaki/2/#c5869
#PBS -l select=1:ngpus=1
#PBS -M skmkt3a2o1i@gmail.com -m be
#PBS -N diseg-pretrain

module load cuda/12.1
source /home/s2210411/conda/etc/profile.d/conda.sh
conda activate diseg-py38
cd ${PBS_O_WORKDI}
# export PYTHONPATH="/home/s2210411/{}:$PYTHONPATH"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MUSTC_ROOT=/home/s2210411/Data/MuST-C-v1.0
LANG=de

PRETRAIN_DIR=/home/s2210411/DiSeg/pretrain_dir
W2V_MODEL=/home/s2210411/DiSeg/w2v/wav2vec_small.pt


python /home/s2210411/DiSeg/train.py ${MUSTC_ROOT}/en-${LANG} --text-data ${MUSTC_ROOT}/data-bin/mustc_en_${LANG}_text --tgt-lang ${LANG} --ddp-backend=legacy_ddp \
  --config-yaml config_raw.yaml \
  --train-subset train \
  --valid-subset dev \
  --save-dir ${PRETRAIN_DIR} \
  --max-tokens 2000000  --max-tokens-text 8192 \
  --update-freq 1 \
  --task speech_to_text_multitask \
  --criterion speech_to_text_multitask \
  --label-smoothing 0.1 \
  --arch convtransformer_espnet_base_wav2vec \
  --w2v2-model-path ${W2V_MODEL} \
  --optimizer adam \
  --lr 2e-3 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 8000 \
  --clip-norm 10.0 \
  --seed 1 \
  --ext-mt-training \
  --eval-task ext_mt \
  --eval-bleu \
  --eval-bleu-args '{"beam": 1,"prefix_size":1}' \
  --eval-bleu-print-samples \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  --keep-best-checkpoints 10 \
  --save-interval-updates 1000 \
  --keep-interval-updates 15 \
  --max-source-positions 800000 \
  --skip-invalid-size-inputs-valid-test \
  --dropout 0.1 --activation-dropout 0.1 --attention-dropout 0.1 --layernorm-embedding \
  --empty-cache-freq 1000 \
  --ignore-prefix-size 1 \
  --patience 10 \
  --fp16 
