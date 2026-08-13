
JOB_NAME='action_finetune_f8x256_70e_k400'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PREFIX='/home/ndr/Container/ImageDataset/Kinetics-400'
DATA_PATH='/home/ndr/Container/ImageDataset/Kinetics-400'

python export_onnx.py \
        --model codat_act_s \
        --data_path ${DATA_PATH} \
        --prefix ${PREFIX} \
        --data_set 'Kinetics_sparse' \
        --split ',' \
        --nb_classes 400 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 86 \
        --num_sample 1 \
        --input_size 256 \
        --short_side_size 256 \
        --save_ckpt_freq 100 \
        --input_shape 'NCHW' \
        --num_frames 8 \
        --num_workers 8 \
        --test_num_segment 4 \
        --test_num_crop 3 \
        --dist_eval \
        --test_best \
        --bf16
