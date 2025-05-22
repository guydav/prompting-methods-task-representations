#!/bin/bash
datasets=('antonym')
# datasets=('antonym' 'capitalize' 'country-capital' 'english-french' 'present-past' 'singular-plural')
# datasets=('capitalize' 'country-capital' 'english-french' 'present-past' 'singular-plural')
cd ../function_vectors

# model_name="openai-community/gpt2-xl"
# model_name="gpt2-xl"
# model_name="EleutherAI/gpt-j-6b"
# model_name="meta-llama/Llama-2-7b-hf"
model_name="meta-llama/Llama-2-70b-hf"

for d_name in "${datasets[@]}"
do
    echo "Running Script for: ${d_name}"
    python evaluate_function_vector.py --dataset_name="${d_name}"  --model_name="${model_name}"
done
