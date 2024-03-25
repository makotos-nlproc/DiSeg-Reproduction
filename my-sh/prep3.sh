export PYTHONPATH="/home/s2210411/DiSeg/exapmles:$PYTHONPATH"
MUSTC_ROOT=/home/s2210411/Data/MuST-C-v1.0
LANG=de

# 3. prepare mustc mt data
MUSTC_TEXT_ROOT=${MUSTC_ROOT}/en-${LANG}-text
SPM_MODEL=${MUSTC_ROOT}/en-${LANG}/spm_unigram10000_raw.model
mkdir ${MUSTC_TEXT_ROOT}

for SPLIT in train dev tst-COMMON
do
    for L in en ${LANG}
    do
        python3 examples/speech_to_text/apply_spm.py \
            --input-file ${MUSTC_ROOT}/en-${LANG}/data/${SPLIT}/txt/${SPLIT}.${L} \
            --output-file ${MUSTC_TEXT_ROOT}/${SPLIT}.spm.${L}  \
            --model ${SPM_MODEL}
    done
done

fairseq-preprocess --source-lang en --target-lang ${LANG} \
    --trainpref ${MUSTC_TEXT_ROOT}/train.spm --validpref ${MUSTC_TEXT_ROOT}/dev.spm \
    --testpref ${MUSTC_TEXT_ROOT}/tst-COMMON.spm \
    --destdir ${MUSTC_ROOT}/data-bin/mustc_en_${LANG}_text \
    --tgtdict ${MUSTC_ROOT}/en-${LANG}/spm_unigram10000_raw.txt \
    --srcdict ${MUSTC_ROOT}/en-${LANG}/spm_unigram10000_raw.txt \
    --nwordssrc 10000 --nwordstgt 10000 \
    --workers 60