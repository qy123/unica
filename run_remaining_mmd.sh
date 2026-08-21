#!/bin/bash
cd /workspace/jinjin/UniCA-master/UniCA-master
source .venv/bin/activate
export DATA_PATH=/workspace/jinjin/UniCA-master/UniCA-master/data/
export MODEL_PATH=/workspace/jinjin/UniCA-master/UniCA-master/models/unica/
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

for dataset in Climate Energy Environment Public_Health Security SocialGood; do
    echo "========================================"
    echo "Starting: time-mmd/$dataset"
    echo "========================================"
    
    python main.py       --model_name ts_adapter/unica       --base_model chronos_bolt_base       --datasets time-mmd/$dataset       --num_workers 0       --num_batches_per_epoch 20       --max_epochs 100       --indexed_sample       --split_val       --gradient_clip 1.0       --batch_size 32       --lr 1e-5       --d_multi_modal 32       --dropout 0.5       --weight_decay 0.1       --with_future       --future_with_gate       --only_quantile_loss       --normalized_loss       --sample_output       --save_ckpt
    
    echo "Finished: time-mmd/$dataset"
    echo ""
done
