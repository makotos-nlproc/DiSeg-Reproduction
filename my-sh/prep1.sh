export PYTHONPATH="/home/s2210411/DiSeg/exapmles:$PYTHONPATH"
MUSTC_ROOT=/home/s2210411/Data/MuST-C-v1.0
LANG=de

# 1. prepare raw mustc data
python3 examples/speech_to_text/prep_mustc_data_raw.py \
    --data-root ${MUSTC_ROOT} --tgt-lang ${LANG}