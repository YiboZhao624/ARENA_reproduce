#!/bin/bash

accelerate launch \
  --config_file                  ../configs/deepspeed_zero2.yaml \
  ../src/sft.py \
  --model_name_or_path           ../model/Qwen2.5-7B-Instruct \
  --dataset_name                 ../data/data_train/sft/sft_25000.jsonl \
  --per_device_train_batch_size  4 \
  --output_dir                   ../checkpoints/Qwen2.5-7B-Instruct-SFT_25000 \
  --bf16                         True \
  --gradient_accumulation_steps  8 \
  --num_train_epochs             1 \
  --logging_steps                1 \
  --eval_strategy                steps \
  --eval_steps                   100 \
  --learning_rate                1e-5 \
  --max_grad_norm                0.3 \
  --warmup_ratio                 0.1 \
  --torch_dtype                  bfloat16 \
  --gradient_checkpointing       True
