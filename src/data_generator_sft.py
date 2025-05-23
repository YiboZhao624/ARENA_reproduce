import json
import os
import random
import argparse

def load_and_process_data(data_path, prompt_template_first_step, prompt_template_second_step, limit, name):
    data = []

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)

    print(f'Loaded data from {data_path}, total: {len(data)}, limit: {limit}')

    random.shuffle(data)
    data = data[:limit]

    processed_data = []

    # Statistics for supporting_facts length distribution
    supporting_facts_lens = {i: 0 for i in range(1, 11)}

    for item in data:
        question = item['question']
        answers = item['answer_labels']

        # Retrieve context passages
        ctx = item['metadata']['retrieval_contexts']
        k = len(ctx)
        if k == 1:
            context = ctx[0]['contents']
        else:
            context = '\n'.join([f'{i + 1}. {c["contents"]}' for i, c in enumerate(ctx[:k])])

        # Retrieve supporting facts and match them to context indices
        supporting_facts = item['metadata']['supporting_facts']
        supporting_ids = []
        for supporting_fact in supporting_facts:
            content = supporting_fact['contents']
            support_id = -1
            for i, c in enumerate(ctx):
                if content in c['contents']:
                    support_id = i + 1
                    break
            if support_id == -1:
                print('Error: supporting content not found in contexts')
            supporting_ids.append(support_id)

        supporting_ids = sorted(set(supporting_ids))
        supporting_facts_lens[len(supporting_ids)] += 1

        relevance_ids = str(supporting_ids)
        answer = random.choice(answers)
        prompt_first_step = prompt_template_first_step.format(question=question, references=context, relevance_ids=relevance_ids, answer=answer)

        prompt_second_step = prompt_template_second_step.format(question=question, references=context)
        question_type = item['question_type']

        processed_data.append({
            'prompt_first_step': prompt_first_step,
            'prompt_second_step': prompt_second_step,
            'answer': answer,
            'question_type': question_type,
            'supporting_ids': supporting_ids,
            'supporting_facts': supporting_facts,
            'name': name,
        })

    print(f'Supporting facts length distribution: {supporting_facts_lens}')
    return processed_data

def make_dialogs(data):
    dialogs = []
    for item in data:
        dialog = {
            "prompt": [{"role": "user", "content": item['prompt_first_step']}],
            "prompt_second_step": [{"role": "user", "content": item['prompt_second_step']}],
            "answer": item['answer'],
            "supporting_ids": item.get('supporting_ids', []),
            "name": item['name'],
        }
        dialogs.append(dialog)
    
    return dialogs

# for reasoning
prompt_template_first_step = '''Generate a thinking process showing how to derive the given answer using ONLY the specified relevance IDs from provided references.
**Question**:
{question}

**References**:
{references}

**Relevance IDs**:
{relevance_ids}

**Answer**:
{answer}
'''

# second step
prompt_template_second_step = '''A conversation between User and Assistant. The user asks a question and give some references. The assistant should answer the question based on the references. 
User's input will always contain:

<question>
[The question to answer]
</question>

<references>
[References starting with numbers]
</references>

Assistant's response must contain EXACTLY three sections:

<relevance>
[List ONLY reference numbers that provide useful information in square brackets, e.g. [1,5]]
</relevance>

<analysis>
[Combine information from relevant references to build the answer. Explicitly mention which references support each claim]
</analysis>

<answer>
[Answer with ONLY a short phrase or single word. No explanations]
</answer>

**User**:

<question>
{question}
</question>

<references>
{references}
</references>
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='hotpotqa', help='Dataset name')
    parser.add_argument('--train-limit', type=int, default=10000, help='Max number of training examples to process')
    parser.add_argument('--test-limit', type=int, default=500, help='Max number of testing examples to process')
    args = parser.parse_args()

    name = args.name
    train_limit = args.train_limit
    test_limit = args.test_limit

    train_datapath = f'../data/original/{name}/train.jsonl'
    test_datapath = f'../data/original/{name}/dev.jsonl'

    random.seed(123456)

    test_data = load_and_process_data(test_datapath, prompt_template_first_step, prompt_template_second_step, test_limit, name)
    train_data = load_and_process_data(train_datapath, prompt_template_first_step, prompt_template_second_step, train_limit, name)

    test_data = make_dialogs(test_data)
    train_data = make_dialogs(train_data)

    print(f'Train data size: {len(train_data)}')
    print(f'Test data size: {len(test_data)}')

    output_dir = f'../data/data_conversation/{name}'
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'train_sft_first_step.jsonl'), 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(os.path.join(output_dir, 'test_sft_first_step.jsonl'), 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
