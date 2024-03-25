export PYTHONPATH="/home/s2210411/DiSeg/exapmles:$PYTHONPATH"
MUSTC_ROOT=/home/s2210411/Data/MuST-C-v1.0
LANG=de

# 4. generate the wav list and reference file for SimulEval
EVAL_ROOT=${MUSTC_ROOT}/en-de-simuleval # such as ${MUSTC_ROOT}/en-de-simuleval
for SPLIT in dev tst-COMMON
do
    python examples/speech_to_text/seg_mustc_data.py \
    --data-root ${MUSTC_ROOT} --lang ${LANG} \
    --split ${SPLIT} --task st \
    --output ${EVAL_ROOT}/${SPLIT}
done