from gpt_api import openai_call
import json
from tqdm import tqdm
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='hotpotqa', help='Dataset name')
    parser.add_argument('--testdata-name', type=str, default='test', help='Input JSONL test data file name')
    parser.add_argument('--saved-name', type=str, default='test_gpt', help='Output result file name (without extension)')
    args = parser.parse_args()

    name = args.name
    testdata_name = args.testdata_name
    saved_name = args.saved_name

    # Load input data
    testdata_path = f'../data/data_raw/{name}/{testdata_name}.jsonl'
    data = []
    with open(testdata_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    prompts = [item['prompt'] for item in data]

    # Start inference
    start_time = time.time()
    responses = []
    for prompt in tqdm(prompts, desc="Running GPT inference"):
        response = openai_call(
            messages=prompt,
            temperature=0,
            model="o1-2024-12-17"
        )
        responses.append(response)
    print(f'Inference time: {time.time() - start_time:.2f} seconds')

    # Collect results
    results = []
    for i, item in enumerate(data):
        results.append({
            'prompt': prompts[i],
            'model_response': responses[i],
            'answers': item['answers'],
            'supporting_ids': item.get('supporting_ids', []),
        })

    # Save results
    import os
    output_dir = f'../results/gpt_inference_results/{name}'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{saved_name}.jsonl')
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f'Saved results to {output_path}')
