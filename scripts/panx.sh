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
USE_SYNTAX=${2:-"false"}
SEED=${3:-1111}

DATA_DIR=${HOME_DIR}/download
OUT_DIR=${HOME_DIR}/outputs
DATA_DIR=$DATA_DIR/panx_udpipe_processed

export CUDA_VISIBLE_DEVICES=$GPU
LANGS="en,ar,bg,de,el,es,fr,hi,ru,tr,ur,vi,ko,nl,pt"

if [[ "$USE_SYNTAX" == 'true' ]]; then
    SAVE_DIR="${OUT_DIR}/panx/syntax-seed${SEED}"
else
    SAVE_DIR="${OUT_DIR}/panx/seed${SEED}"
fi
mkdir -p $SAVE_DIR;

export PYTHONPATH=$HOME_DIR;
python $HOME_DIR/third_party/ner.py \
    --seed $SEED \
    --data_dir $DATA_DIR \
    --labels $DATA_DIR/labels.txt \
    --model_type bert \
    --model_name_or_path bert-base-multilingual-cased \
    --task_name panx \
    --output_dir $SAVE_DIR \
    --max_seq_length 128 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 1 \
    --per_gpu_train_batch_size 32 \
    --per_gpu_eval_batch_size 32 \
    --save_steps 5000 \
    --learning_rate 2e-5 \
    --do_train \
    --do_predict \
    --predict_langs $LANGS \
    --train_langs en \
    --log_file $SAVE_DIR/train.log \
    --eval_all_checkpoints \
    --eval_patience -1 \
    --overwrite_output_dir \
    --save_only_best_checkpoint \
    --use_syntax $USE_SYNTAX \
    --use_pos_tag $USE_SYNTAX \
    --use_structural_loss $USE_SYNTAX \
    --struct_loss_coeff 0.5 \
    --num_gat_layer 4 \
    --num_gat_head 4 \
    --max_syntactic_distance 1 \
    --num_syntactic_heads 1 \
    --syntactic_layers 0,1,2,3,4,5,6,7,8,9,10,11 \
    2>&1 | tee $SAVE_DIR/output.log;
