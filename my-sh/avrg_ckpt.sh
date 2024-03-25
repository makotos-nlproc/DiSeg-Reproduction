#!/bin/bash
#PBS -j oe
#PBS -q GPU-1A
#PBS -l select=1:ngpus=1
#PBS -M skmkt3a2o1i@gmail.com -m be
#PBS -N itst-st-train
# Nミス

module load cuda/12.1
source /home/s2210411/conda/etc/profile.d/conda.sh
conda activate diseg-py38
cd ${PBS_O_WORKDI}

PRETRAIN_DIR=/home/s2210411/DiSeg/ckpt

python /home/s2210411/DiSeg/scripts/average_checkpoints.py \
    --inputs ${PRETRAIN_DIR} \
    --num-update-checkpoints 10 \
    --output ${PRETRAIN_DIR}/mt_pretrain_model.pt \
    --best True