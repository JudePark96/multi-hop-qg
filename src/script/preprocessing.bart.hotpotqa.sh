#!/bin/bash

export PYTHONPATH="../../src"

MODEL_NAME_OR_PATH='../resources/bart-large-add-tokens'
MODEL_TYPE='bart'
MODEL_SCALE='large'
MAX_SEQ_LENGTH=384
TRAIN_INPUT_EXAMPLES_OUTPUT_PATH="../resources/hotpotqa/preprocessed/${MODEL_TYPE}_${MODEL_SCALE}_train_normal_examples_maxlen${MAX_SEQ_LENGTH}.pkl.gz"
DEV_INPUT_EXAMPLES_OUTPUT_PATH="../resources/hotpotqa/preprocessed/${MODEL_TYPE}_${MODEL_SCALE}_dev_normal_examples_maxlen${MAX_SEQ_LENGTH}.pkl.gz"


python3 data_utils/hotpotqa/feature_util.py --preprocessing_type "normal_input_example" \
                                            --model_name_or_path ${MODEL_NAME_OR_PATH} \
                                            --model_type ${MODEL_TYPE} \
                                            --hotpotqa_path ../resources/hotpotqa/hotpot_train_v1.1.json \
                                            --paragraph_path ../resources/hotpotqa/train/selected_paras_QG.json \
                                            --entity_path ../resources/hotpotqa/train/entities.json \
                                            --output_path ${TRAIN_INPUT_EXAMPLES_OUTPUT_PATH} \
                                            --max_seq_length ${MAX_SEQ_LENGTH}

python3 data_utils/hotpotqa/feature_util.py --preprocessing_type "normal_input_example" \
                                            --model_name_or_path ${MODEL_NAME_OR_PATH} \
                                            --model_type ${MODEL_TYPE} \
                                            --hotpotqa_path ../resources/hotpotqa/hotpot_dev_distractor_v1.json \
                                            --paragraph_path ../resources/hotpotqa/dev/selected_paras_QG.json \
                                            --entity_path ../resources/hotpotqa/dev/entities.json \
                                            --output_path ${DEV_INPUT_EXAMPLES_OUTPUT_PATH} \
                                            --max_seq_length ${MAX_SEQ_LENGTH}

python3 data_utils/hotpotqa/feature_util.py --preprocessing_type 'normal_input_feature' \
                                            --example_path ${TRAIN_INPUT_EXAMPLES_OUTPUT_PATH} \
                                            --model_name_or_path ${MODEL_NAME_OR_PATH} \
                                            --model_type ${MODEL_TYPE} \
                                            --output_path "../resources/hotpotqa/preprocessed/${MODEL_TYPE}_${MODEL_SCALE}_train_normal_features.pkl.gz" \
                                            --max_seq_length ${MAX_SEQ_LENGTH}

python3 data_utils/hotpotqa/feature_util.py --preprocessing_type 'normal_input_feature' \
                                            --example_path ${DEV_INPUT_EXAMPLES_OUTPUT_PATH} \
                                            --model_name_or_path ${MODEL_NAME_OR_PATH} \
                                            --model_type ${MODEL_TYPE} \
                                            --output_path "../resources/hotpotqa/preprocessed/${MODEL_TYPE}_${MODEL_SCALE}_dev_normal_features.pkl.gz" \
                                            --max_seq_length ${MAX_SEQ_LENGTH}
