#!/bin/bash

export PYTHONPATH='../../src'


CONFIG_NAME_OR_PATH='../resources/bart-base-add-tokens'
DEV_FEATURES_PATH='../resources/hotpotqa/preprocessed/bart_dev_normal_features.pkl.gz'
BATCH_SIZE=16
NUM_BEAMS=5
MAX_LENGTH=128
NUM_WORKERS=32

SAVE_CHECKPOINTS_DIR="../checkpoints/hotpotqa/NormalBART/E5_B16_WARM0.06_NORM1.0_LR3e-05/"
CHECKPOINT_BIN="3epoch.pth"
OUTPUT_FILE_NAME="inference_3epoch.json"

export CUDA_VISIBLE_DEVICES=0

python3 ./inference/normal_bart_inference.py --save_checkpoints_dir ${SAVE_CHECKPOINTS_DIR} \
                                           --checkpoint_bin ${CHECKPOINT_BIN} \
                                           --output_file_name ${OUTPUT_FILE_NAME} \
                                           --config_name_or_path ${CONFIG_NAME_OR_PATH} \
                                           --dev_features_path ${DEV_FEATURES_PATH} \
                                           --batch_size ${BATCH_SIZE} \
                                           --num_beams ${NUM_BEAMS} \
                                           --max_length ${MAX_LENGTH} \
                                           --num_workers ${NUM_WORKERS} \
                                           --is_cuda \
                                           --is_early_stopping
