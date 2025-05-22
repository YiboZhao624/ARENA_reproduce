ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ../configs/deepspeed_zero2.yaml \
    --num_processes=7 ../src/grpo.py \
    --config ../configs/grpo.yaml
