#!/bin/bash

CURRENT_DIR=`pwd`
HOME_DIR=`realpath ..`
LANG=(af ar bg de el en es et fi fr he hi hu id it ja ko mr nl pt ru ta te tr ur vi zh)

############################# Downloading UDPipe #############################

URL_PREFIX='https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131'
declare -A LANG_MAP
LANG_MAP['af']='afrikaans-afribooms-ud-2.5-191206.udpipe'
LANG_MAP['ar']='arabic-padt-ud-2.5-191206.udpipe'
LANG_MAP['bg']='bulgarian-btb-ud-2.5-191206.udpipe'
LANG_MAP['de']='german-gsd-ud-2.5-191206.udpipe'
LANG_MAP['el']='greek-gdt-ud-2.5-191206.udpipe'
LANG_MAP['en']='english-ewt-ud-2.5-191206.udpipe'
LANG_MAP['es']='spanish-gsd-ud-2.5-191206.udpipe'
LANG_MAP['et']='estonian-edt-ud-2.5-191206.udpipe'
LANG_MAP['fi']='finnish-tdt-ud-2.5-191206.udpipe'
LANG_MAP['fr']='french-gsd-ud-2.5-191206.udpipe'
LANG_MAP['he']='hebrew-htb-ud-2.5-191206.udpipe'
LANG_MAP['hi']='hindi-hdtb-ud-2.5-191206.udpipe'
LANG_MAP['hu']='hungarian-szeged-ud-2.5-191206.udpipe'
LANG_MAP['id']='indonesian-gsd-ud-2.5-191206.udpipe'
LANG_MAP['it']='italian-isdt-ud-2.5-191206.udpipe'
LANG_MAP['ja']='japanese-gsd-ud-2.5-191206.udpipe'
LANG_MAP['ko']='korean-kaist-ud-2.5-191206.udpipe' # 'korean-gsd-ud-2.5-191206.udpipe'
LANG_MAP['mr']='marathi-ufal-ud-2.5-191206.udpipe'
LANG_MAP['nl']='dutch-alpino-ud-2.5-191206.udpipe'
LANG_MAP['pt']='portuguese-bosque-ud-2.5-191206.udpipe' # 'portuguese-gsd-ud-2.5-191206.udpipe'
LANG_MAP['ru']='russian-gsd-ud-2.5-191206.udpipe'
LANG_MAP['ta']='tamil-ttb-ud-2.5-191206.udpipe'
LANG_MAP['te']='telugu-mtg-ud-2.5-191206.udpipe'
LANG_MAP['tr']='turkish-imst-ud-2.5-191206.udpipe'
LANG_MAP['ur']='urdu-udtb-ud-2.5-191206.udpipe'
LANG_MAP['vi']='vietnamese-vtb-ud-2.5-191206.udpipe'
LANG_MAP['zh']='chinese-gsd-ud-2.5-191206.udpipe'

UDPIPE_DIR=${CURRENT_DIR}/models
mkdir -p $UDPIPE_DIR

for lang in ${LANG[@]}; do
    if [[ ! -f ${UDPIPE_DIR}/${LANG_MAP[${lang}]} ]]; then
        curl -Lo ${UDPIPE_DIR}/${LANG_MAP[${lang}]} ${URL_PREFIX}/${LANG_MAP[${lang}]}
    fi
done

#############################

DATA_DIR=${HOME_DIR}/download/panx
OUT_DIR=${HOME_DIR}/download/panx_udpipe_processed
mkdir -p $OUT_DIR

# train data processing
if [[ ! -f ${OUT_DIR}/train-en.jsonl ]]; then
    python process.py \
        --task panx \
        --input_file ${DATA_DIR}/train-en.tsv \
        --output_file ${OUT_DIR}/train-en.jsonl \
        --pre_lang en \
        --udpipe_model ${UDPIPE_DIR}/${LANG_MAP['en']} \
        --workers 60;
fi

for lang in "${LANG[@]}"; do
    for split in dev test; do
        outfile=${OUT_DIR}/${split}-${lang}.jsonl
        if [[ ! -f $outfile ]]; then
            python process.py \
                --task panx \
                --input_file ${DATA_DIR}/${split}-${lang}.tsv \
                --output_file $outfile \
                --udpipe_model ${UDPIPE_DIR}/${LANG_MAP[${lang}]} \
                --pre_lang $lang \
                --workers 60;
        fi
    done
done
