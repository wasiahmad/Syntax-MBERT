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
SEED=${4:-1111}

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
python $HOME_DIR/third_party/classify.py \
    --seed $SEED \
    --model_type bert \
    --model_name_or_path bert-base-multilingual-cased \
    --train_language en \
    --task_name $TASK \
    --do_train \
    --do_predict \
    --data_dir $DATA_DIR/${TASK}_udpipe_processed \
    --gradient_accumulation_steps 1 \
    --per_gpu_train_batch_size 32 \
    --per_gpu_eval_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --max_seq_length 128 \
    --output_dir $SAVE_DIR/ \
    --save_steps 2000 \
    --eval_all_checkpoints \
    --log_file 'train.log' \
    --predict_languages $LANGS \
    --save_only_best_checkpoint \
    --overwrite_output_dir \
    --use_syntax $USE_SYNTAX \
    --use_pos_tag $USE_SYNTAX \
    --use_structural_loss $USE_SYNTAX \
    --struct_loss_coeff 1.0 \
    --num_gat_layer 4 \
    --num_gat_head 4 \
    --max_syntactic_distance 4 \
    --num_syntactic_heads 1 \
    --syntactic_layers 0,1,2,3,4,5,6,7,8,9,10,11 \
    2>&1 | tee $SAVE_DIR/output.log;
