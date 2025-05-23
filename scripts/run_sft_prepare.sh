#!/bin/bash

# Define arrays for name and corresponding train_limit
names=("hotpotqa" "two_wiki" "musique")
train_limits=(10000 10000 5000)
test_limit=500

# Iterate over datasets
for i in "${!names[@]}"; do
  name=${names[$i]}
  train_limit=${train_limits[$i]}
  
  echo "Running sft preparation for $name..."
  python ../src/data_generator_sft.py --name "$name" --train-limit "$train_limit" --test-limit "$test_limit"
  python ../src/vllm_inference_sft.py --name "$name" --testdata-name "train_sft_first_step" --saved-name "train_sft_first_step" --model-path "../model/Qwen2.5-7B-Instruct"

done

python ../src/datamaker_sft.py --name "hotpotqa" "two_wiki" "musique" --trainfile-name "train_sft_first_step" --saved-name "sft_25000"
