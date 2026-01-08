NCCL_P2P_DISABLE="1" \
NCCL_IB_DISABLE="1" \
WANDB_API_KEY=49d8431442798cf902aee642df7b8f1cdb16e184 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=1258291 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model /opt/data/private/hyp/models/Qwen/Qwen2.5-VL-3B-Instruct \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs gui_r1_combined \
    --custom_register_path examples/custom/gui-r1_3k.py \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.8 \
    --vllm_tensor_parallel_size 1 \
    --vllm_enforce_eager \
    --offload_optimizer \
    --offload_model \
    --sleep_level 1 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset 'gui-r1' \
    --val_dataset 'gui-r1-val' \
    --load_from_cache_file true \
    --max_length 2048 \
    --max_pixels 1258291 \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 8 \
    --save_strategy 'steps' \
    --eval_strategy 'steps' \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --output_dir output/GUI_R1 \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero3 \
    --log_completions true \
    --report_to wandb \
    --num_iterations 1 \
    --async_generate false \
    --beta 0.001 