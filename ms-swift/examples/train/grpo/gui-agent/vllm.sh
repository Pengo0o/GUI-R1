CUDA_VISIBLE_DEVICES=6,7 \
swift rollout \
    --model /opt/data/private/hyp/models/Qwen/Qwen2.5-VL-3B-Instruct \
    --vllm_data_parallel_size 2