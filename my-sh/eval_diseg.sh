#!/bin/bash

module load cuda/12.1
source /home/s2210411/conda/etc/profile.d/conda.sh
conda activate diseg-py38

export PYTHONPATH="/home/s2210411/DiSeg/src/simuleval:$PYTHONPATH"
export PYTHONPATH="/home/s2210411/DiSeg:$PYTHONPATH"
export PYTHONPATH="/home/s2210411/DiSeg/examples:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

MUSTC_ROOT=/home/s2210411/Data/MuST-C-v1.0
LANG=de
EVAL_ROOT=/home/s2210411/Data/MuST-C-v1.0/en-de-simuleval/
SAVE_DIR=/home/s2210411/DiSeg/ckpt
OUTPUT_DIR=/home/s2210411/DiSeg/simuleval_results/v0

# python /home/s2210411/DiSeg/scripts/average_checkpoints.py \
#     --inputs ${SAVE_DIR} \
#     --num-update-checkpoints 5 \
#     --output ${SAVE_DIR}/average-model.pt \
#     --best True

lagging_seg=5 # lagging segment in DiSeg

simuleval --agent /home/s2210411/DiSeg/diseg_agent.py \
    --source ${EVAL_ROOT}/tst-COMMON/tst-COMMON.wav_list \
    --target ${EVAL_ROOT}/tst-COMMON/tst-COMMON.${LANG} \
    --data-bin ${MUSTC_ROOT}/en-${LANG} \
    --config config_raw.yaml \
    --model-path ${SAVE_DIR}/average-model.pt \
    --output ${OUTPUT_DIR} \
    --lagging-segment ${lagging_seg}  \
    --lang ${LANG} \
    --scores --gpu --fp16 \
    --port 12345