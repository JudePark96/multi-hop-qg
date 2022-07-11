#!/bin/bash

EPOCH=4
PREDICT_DATA_PATH=../resources/totto/totto_dev_data.jsonl
MODEL_PATH="./desc_generation/models/t5-large/checkpoint_ep${EPOCH}.pt"
PREDICT_PATH="./desc_generation/models/t5-large/generation_ep${EPOCH}.txt"

export CUDA_VISIBLE_DEVICES=4

python3 ./desc_generation/table_desc_gen_train.py \
                                     --do_predict \
                                     --model_name "google/t5-v1_1-large" \
                                     --predict_batch_size 8 \
                                     --max_len 512 \
                                     --predict_file ${PREDICT_DATA_PATH} \
                                     --predict_path ${PREDICT_PATH} \
                                     --checkpoint_model_path ${MODEL_PATH}
