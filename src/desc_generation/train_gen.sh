SOCKETS=$(lscpu | grep Socket | awk '{print $2}')
TOTAL_CORES=`expr $CORES \* $SOCKETS`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
KMP_BLOCKTIME=1

export OMP_NUM_THREADS=48
export $KMP_SETTING
export KMP_BLOCKTIME=$KMP_BLOCKTIME

echo -e "### using OMP_NUM_THREADS=$TOTAL_CORES"
echo -e "### using $KMP_SETTING"
echo -e "### using KMP_BLOCKTIME=$KMP_BLOCKTIME\n"
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NODE_RANK=0
export N_NODES=1
export N_GPU_NODE=4
export WORLD_SIZE=4
export MASTER_PORT=$(expr $RANDOM + 1000)
export MASTER_ADDR="localhost"

DEVICE_IDS=0,1,2,3
RUN_ID=0
TRAIN_DATA_PATH=../resources/totto/totto_train_data.jsonl
MODEL_PATH=./desc_generation/models/t5-large

python3 -m torch.distributed.launch \
          --nproc_per_node ${N_GPU_NODE} \
          --nnodes ${N_NODES} \
          --node_rank ${NODE_RANK} \
          --master_addr ${MASTER_ADDR} \
          --master_port "${MASTER_PORT}" \
          ./desc_generation/table_desc_gen_train.py \
                                     --do_train \
                                     --n_gpu ${WORLD_SIZE} \
                                     --device_ids ${DEVICE_IDS} \
                                     --prefix ${RUN_ID} \
                                     --model_name "google/t5-v1_1-large" \
                                     --per_gpu_train_batch_size 4 \
                                     --max_len 512 \
                                     --num_train_epochs 5 \
                                     --accumulate_gradients 1 \
                                     --gradient_accumulation_steps 1 \
                                     --warmup_ratio 0.06 \
                                     --train_file ${TRAIN_DATA_PATH} \
                                     --output_dir ${MODEL_PATH}