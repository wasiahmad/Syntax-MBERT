#!/bin/bash

CURRENT_DIR=`pwd`
HOME_DIR=`realpath ..`
LANG=(en fr es de zh ja ko)

############################# Downloading UDPipe #############################

URL_PREFIX='https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131'
declare -A LANG_MAP
LANG_MAP['en']='english-ewt-ud-2.5-191206.udpipe'
LANG_MAP['fr']='french-gsd-ud-2.5-191206.udpipe'
LANG_MAP['es']='spanish-gsd-ud-2.5-191206.udpipe'
LANG_MAP['de']='german-gsd-ud-2.5-191206.udpipe'
LANG_MAP['zh']='chinese-gsd-ud-2.5-191206.udpipe'
LANG_MAP['ja']='japanese-gsd-ud-2.5-191206.udpipe'
LANG_MAP['ko']='korean-gsd-ud-2.5-191206.udpipe'

OUT_DIR=${CURRENT_DIR}/models
mkdir -p $OUT_DIR

for lang in ${LANG[@]}; do
    if [[ ! -f ${OUT_DIR}/${LANG_MAP[${lang}]} ]]; then
        curl -Lo ${OUT_DIR}/${LANG_MAP[${lang}]} ${URL_PREFIX}/${LANG_MAP[${lang}]}
    fi
done

#############################

DATA_DIR=${HOME_DIR}/download/pawsx
OUT_DIR=${HOME_DIR}/download/pawsx_udpipe_processed
mkdir -p $OUT_DIR

# train data processing
python process.py \
    --task pawsx \
    --input_file ${DATA_DIR}/train-en.tsv \
    --output_file ${OUT_DIR}/train-en.jsonl \
    --pre_lang en \
    --hyp_lang en \
    --workers 60;

for split in dev test; do
    for lang in "${LANG[@]}"; do
        python process.py \
            --task pawsx \
            --input_file ${DATA_DIR}/${split}-${lang}.tsv \
            --output_file ${OUT_DIR}/${split}-${lang}.jsonl \
            --pre_lang $lang \
            --hyp_lang $lang \
            --workers 60;
    done
done
