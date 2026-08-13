export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

JOB_NAME='action_finetune_f8x256_70e_k400'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PREFIX='/home/dsdl-4090/Documents/dsdl_ssd/ImageDataset/Kinetics-400'
DATA_PATH='/home/dsdl-4090/Documents/dsdl_ssd/ImageDataset/Kinetics-400'

GPUS=4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --nproc_per_node=$GPUS \
        --master_port=$PORT \
        run_class_finetuning.py \
        --model codat_act_s \
        --data_path ${DATA_PATH} \
        --prefix ${PREFIX} \
        --data_set 'Kinetics_sparse' \
        --split ',' \
        --nb_classes 400 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 64 \
        --num_sample 2 \
        --input_size 256 \
        --short_side_size 256 \
        --save_ckpt_freq 100 \
        --input_shape 'NCHW' \
        --num_frames 8 \
        --num_workers 12 \
        --warmup_epochs 5 \
        --tubelet_size 1 \
        --epochs 70 \
        --lr 2e-4 \
        --drop_path 0.05 \
        --fc_drop_rate 0.05 \
        --aa rand-m5-n2-mstd0.25-inc1 \
        --opt adamw \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.035 \
        --test_num_segment 4 \
        --test_num_crop 3 \
        --dist_eval \
        --test_best \
        --bf16
