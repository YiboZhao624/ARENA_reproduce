import json
import argparse
import random
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='hotpotqa', help='Dataset name')
    parser.add_argument('--testfile-name', type=str, default='test', help='Test file name')
    args = parser.parse_args()

    name = args.name
    testfile_name = args.testfile_name

    # Load test data from JSONL file
    data_path = f'../data/data_direct/{name}/{testfile_name}.jsonl'
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)

    # Set random seed for reproducibility
    random.seed(123456)

    # Format data into dialog format
    dialogs = []
    for item in data:
        dialog = {
            "prompt": [{"role": "user", "content": item['prompt']}],
            "answers": item['answers'],
            "supporting_ids": item.get('supporting_ids', []),
            "name": item['name'],
        }
        dialogs.append(dialog)

    print(f'Number of dialogs: {len(dialogs)}')

    # Ensure output directory exists
    output_dir = f'../data/data_conversation/{name}'
    os.makedirs(output_dir, exist_ok=True)

    # Save formatted dialogs to a new JSONL file
    output_path = os.path.join(output_dir, f'{testfile_name}.jsonl')
    with open(output_path, 'w', encoding='utf-8') as f:
        for dialog in dialogs:
            f.write(json.dumps(dialog, ensure_ascii=False) + '\n')
