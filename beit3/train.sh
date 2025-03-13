export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
torchrun -m torch.distributed.launch --nproc_per_node=3 --master_port=29501 run_beit3_finetuning.py \
        --model beit3_base_patch16_480 \
        --input_size 480 \
        --task vqav2 \
        --batch_size 4 \
        --num_workers 10 \
        --layer_decay 1.0 \
        --lr 3e-5 \
        --update_freq 1 \
        --epochs 5 \
        --warmup_epochs 1 \
        --drop_path 0.1 \
        --nb_classes 108 \
        --sentencepiece_model /workspace/models/beit3.spm \
        --finetune /workspace/models/beit3_base_patch16_480_vqa.pth \
        --data_path /workspace/vqa-info-data \
        --root_folder /workspace/vqa_dataset/images \
        --output_dir output \
        --log_dir log \
        --weight_decay 0.01 \
        --seed 42 \
        --save_ckpt_freq 1 \
        --task_head_lr_weight 20 \
        --opt_betas 0.9 0.98 \
        --enable_deepspeed