import argparse
import json
from tqdm import tqdm
import os
from gpt_api import openai_call
from evaluate import extract_answer

def format_prompt(model_response, correct_answer):
    prompt = '*************Consider a knowledge Q&A RAG task to test the capability of a testing model, the correct answer list is:*************\n' + correct_answer
    prompt += '\n\n\n\n*************Here is the model\'s response:*************\n' + model_response
    prompt += '\n\n\n\n*************Please check if the model\'s answer is correct. As long as the model\'s answer hits any item (or synonym) in the correct answer list, it can be considered correct. You only need to answer "yes" or "no".*************'
    return prompt

def get_score(response):
    response = response.lower()
    if 'yes' in response:
        return 1
    elif 'no' in response:
        return 0
    else:
        return 0.5

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='hotpotqa')
    parser.add_argument('--result-name', type=str, default='test')
    parser.add_argument('--limit', type=int, default=10000000)
    args = parser.parse_args()


    name = args.name
    result_name = args.result_name
    limit = args.limit

    in_file = f'../results/vllm_inference_results/{name}/{result_name}.jsonl'
    output_dir = f'../results/eval_gpt/{name}'
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f'{result_name}.json')

    data = []
    with open(in_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    data = data[:limit]

    correct = 0
    total = 0
    responses = []
    scores = []
    prompts = []
 
    for item in tqdm(data):
        try:
            model_response = item['model_response']
            model_response = extract_answer(model_response)

            if model_response == '':
                score = 0.0
            else:
                correct_answer = item['answers']
                if isinstance(correct_answer, list):
                    correct_answer = str(correct_answer)

                prompt = format_prompt(model_response, correct_answer)
                prompts.append(prompt)

                response = openai_call(prompt, temperature=0)
                responses.append(response)

                score = get_score(response)
            scores.append(score)
        except Exception as e:
            print(e)
            prompt = 'Error'
            prompts.append(prompt)
            response = 'Error'
            responses.append(response)
            score = 0.0
            scores.append(score)

        correct += score
        total += 1

        if total % 10 == 0:
            print(correct, total, correct / total)
    print(correct, total, correct / total)

    debug = True
    with open(out_file, 'w') as f:
        data = {
            'correct': correct,
            'total': total,
            'accuracy': correct / total,
            'scores': scores,
        }
        if debug:
            data['prompts'] = prompts
            data['responses'] = responses
        json.dump(data, f, indent=4)
