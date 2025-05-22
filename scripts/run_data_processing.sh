#!/bin/bash

# Define arrays for name and corresponding train_limit
names=("hotpotqa" "two_wiki" "musique")
train_limits=(10000 10000 5000)
test_limit=500

# Iterate over datasets
for i in "${!names[@]}"; do
  name=${names[$i]}
  train_limit=${train_limits[$i]}
  
  echo "Running data processing for $name..."
  python ../src/data_generator.py --name "$name" --train-limit "$train_limit" --test-limit "$test_limit"
  python ../src/datamaker_conversation.py --name "$name" --testfile-name "test"
done

python ../src/datamaker_grpo.py --name "hotpotqa" "two_wiki" "musique" --trainfile-name "train"
