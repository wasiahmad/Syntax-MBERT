#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

CURRENT_DIR=`pwd`
HOME_DIR=`realpath ..`

GPU=${1:-0}
TASK=${2:-"xnli"}
USE_SYNTAX=${3:-"false"}
SEED=${3:-1111}

DATA_DIR=${HOME_DIR}/download
OUT_DIR=${HOME_DIR}/outputs

export CUDA_VISIBLE_DEVICES=$GPU

if [[ "$TASK" == 'xnli' ]]; then
    LANGS="ar,bg,de,el,en,es,fr,hi,ru,tr,ur,vi,zh"
else
    LANGS="de,en,es,fr,ja,ko,zh"
fi

if [[ "$USE_SYNTAX" == 'true' ]]; then
    SAVE_DIR="${OUT_DIR}/${TASK}/syntax-seed${SEED}"
else
    SAVE_DIR="${OUT_DIR}/${TASK}/seed${SEED}"
fi
mkdir -p $SAVE_DIR;

export PYTHONPATH=$HOME_DIR;
python $HOME_DIR/third_party/gxlt_classify.py \
    --seed $SEED \
    --model_type bert \
    --model_name_or_path bert-base-multilingual-cased \
    --task_name $TASK \
    --do_predict \
    --data_dir $DATA_DIR/${TASK}_udpipe_processed \
    --per_gpu_eval_batch_size 32 \
    --max_seq_length 128 \
    --output_dir $SAVE_DIR \
    --log_file 'gxl.log' \
    --use_syntax $USE_SYNTAX \
    --use_pos_tag $USE_SYNTAX \
    --predict_languages $LANGS \
    --overwrite_output_dir \
    2>&1 | tee $SAVE_DIR/output.log;
