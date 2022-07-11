#!/bin/bash


CONFIG_NAME_OR_PATH='../resources/fid-t5-base-add-tokens'
DEV_FEATURES_PATH='../resources/hotpotqa/preprocessed/t5_dev_features.pkl.gz'
BATCH_SIZE=8
NUM_BEAMS=5
MAX_LENGTH=128
NUM_WORKERS=32

SAVE_CHECKPOINTS_DIR="../checkpoints/hotpotqa/FiDT5/E5_B16_WARM0.06_NORM1.0_LR3e-05/"
CHECKPOINT_BIN="4epoch.pth"
OUTPUT_FILE_NAME="inference_4epoch.json"

export CUDA_VISIBLE_DEVICES=4

python3 ./inference/fid_t5_inference.py --save_checkpoints_dir ${SAVE_CHECKPOINTS_DIR} \
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
