#!/bin/bash
#PBS -j oe
#PBS -q GPU-S
# https://www.jaist.ac.jp/iscenter/mpc/kagayaki/2/#c5869
#PBS -l select=1:ngpus=1
#PBS -M skmkt3a2o1i@gmail.com -m be
#PBS -N diseg-train-without-pretrain

module load cuda/12.1
source /home/s2210411/conda/etc/profile.d/conda.sh
conda activate diseg-py38
cd ${PBS_O_WORKDI}
# export PYTHONPATH="/home/s2210411/{}:$PYTHONPATH"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MUSTC_ROOT=/home/s2210411/Data/MuST-C-v1.0
LANG=de

SAVE_DIR==/home/s2210411/ckpt-1
W2V_MODEL=/home/s2210411/DiSeg/w2v/wav2vec_small.pt

mean=0
var=3

# (optional) pre-train a mt encoder/decoder and load the pre-trained model with --load-pretrained-mt-encoder-decoder-from ${PRETRAIN_DIR}/mt_pretrain_model.pt
python /home/s2210411/DiSeg/train.py ${MUSTC_ROOT}/en-${LANG}  --tgt-lang ${LANG} --ddp-backend=legacy_ddp \
  --config-yaml config_raw.yaml \
  --train-subset train_raw \
  --valid-subset dev_raw \
  --save-dir ${SAVE_DIR} \
  --max-tokens 1500000  --batch-size 32 --max-tokens-text 4096 \
  --update-freq 1 \
  --num-workers 8 \
  --task speech_to_text_multitask \
  --criterion speech_to_text_multitask_with_seg \
  --report-accuracy \
  --arch convtransformer_espnet_base_wav2vec_seg \
  --w2v2-model-path ${W2V_MODEL} \
  --optimizer adam \
  --lr 0.0001 \
  --lr-scheduler inverse_sqrt \
  --weight-decay 0.0001 \
  --label-smoothing 0.1 \
  --warmup-updates 4000 \
  --clip-norm 10.0 \
  --seed 1 \
  --seg-encoder-layers 6 \
  --noise-mean ${mean} --noise-var ${var} \
  --st-training --mt-training --asr-training \
  --seg-speech --add-speech-seg-text-ctr \
  --eval-task st \
  --eval-bleu \
  --eval-bleu-args '{"beam": 1,"prefix_size":1}' \
  --eval-bleu-print-samples \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  --keep-best-checkpoints 20 \
  --save-interval-updates 1000 \
  --keep-interval-updates 30 \
  --max-source-positions 800000 \
  --skip-invalid-size-inputs-valid-test \
  --dropout 0.1 --activation-dropout 0.1 --attention-dropout 0.1 --layernorm-embedding \
  --empty-cache-freq 1000 \
  --ignore-prefix-size 1 \
  --fp16 
