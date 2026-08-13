DATA_PATH='/home/dsdl-4090/Documents/dsdl_ssd/ImageDataset/ImageNet-1K'
CODE_PATH='/home/dsdl-4090/Documents/dsdl_ssd/ParFormer_workspace' # modify code path here


ALL_BATCH_SIZE=2048
NUM_GPU=4
NUM_WORKERS=10
GRAD_ACCUM_STEPS=4 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU


NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port 12345 --use_env main.py --data-path $DATA_PATH --batch-size $BATCH_SIZE \
--model CoDAT_S --dist-eval --weight-decay 0.035 --num_workers $NUM_WORKERS --input-size 256 --output_dir ./output/par_s1_relu_221 --enable_wandb --project=New_Parformer 
