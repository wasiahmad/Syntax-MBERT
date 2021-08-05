#!/bin/bash

CURRENT_DIR=`pwd`
HOME_DIR=`realpath ..`
LANG=(en es fr de hi)

############################# Downloading UDPipe #############################

URL_PREFIX='https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131'
declare -A LANG_MAP
LANG_MAP['en']='english-ewt-ud-2.5-191206.udpipe'
LANG_MAP['fr']='french-gsd-ud-2.5-191206.udpipe'
LANG_MAP['es']='spanish-gsd-ud-2.5-191206.udpipe'
LANG_MAP['de']='german-gsd-ud-2.5-191206.udpipe'
LANG_MAP['hi']='hindi-hdtb-ud-2.5-191206.udpipe'

OUT_DIR=${CURRENT_DIR}/models
mkdir -p $OUT_DIR

for lang in ${LANG[@]}; do
    if [[ ! -f ${OUT_DIR}/${LANG_MAP[${lang}]} ]]; then
        curl -Lo ${OUT_DIR}/${LANG_MAP[${lang}]} ${URL_PREFIX}/${LANG_MAP[${lang}]}
    fi
done

#############################

DATA_DIR=${HOME_DIR}/download/mtop
OUT_DIR=${HOME_DIR}/download/mtop_udpipe_processed
mkdir -p $OUT_DIR

# train data processing
if [[ ! -f ${OUT_DIR}/train-en.jsonl ]]; then
    python process.py \
        --task mtop \
        --input_file ${DATA_DIR}/train-en.jsonl \
        --output_file ${OUT_DIR}/train-en.jsonl \
        --pre_lang en \
        --workers 60;
fi

for lang in "${LANG[@]}"; do
    for split in dev test; do
        outfile=${OUT_DIR}/${split}-${lang}.jsonl
        if [[ ! -f $outfile ]]; then
            python process.py \
                --task mtop \
                --input_file ${DATA_DIR}/${split}-${lang}.jsonl \
                --output_file $outfile \
                --pre_lang $lang \
                --workers 60;
        fi
    done
done
