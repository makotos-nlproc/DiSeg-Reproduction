export PYTHONPATH="/home/s2210411/DiSeg/exapmles:$PYTHONPATH"
MUSTC_ROOT=/home/s2210411/Data/MuST-C-v1.0
LANG=de


# 2. prepare vocabulary
python3 examples/speech_to_text/prep_vocab.py \
    --data-root ${MUSTC_ROOT} \
    --vocab-type unigram --vocab-size 10000 --joint \
    --tgt-lang ${LANG}