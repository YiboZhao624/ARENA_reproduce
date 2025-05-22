import re
import json
import string
from collections import Counter
import argparse
import os

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()

def exact_match_score(prediction, ground_truths):
    for ground_truth in ground_truths:
        if normalize_answer(prediction) == normalize_answer(ground_truth):
            return 1.0
    return 0.0

def f1_score(prediction, ground_truths):
    f1_score = 0.0
    for ground_truth in ground_truths:
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_score = max(f1_score, f1)
    return f1_score

def format_score(prediction):
    pattern = r"""
        ^
        <relevance>    [^<]*  </relevance>
        \s*
        <analysis>     [^<]*  </analysis>
        \s*
        <answer>       [^<]*  </answer>
        $
    """
    match = re.match(pattern, prediction, re.DOTALL | re.VERBOSE | re.MULTILINE)
    return 1.0 if match else 0.0

def extract_answer(prediction):
    pattern = r"<answer>\s*(.*?)\s*</answer>"
    match = re.search(pattern, prediction, re.DOTALL | re.MULTILINE)
    if match:
        return match.group(1)
    else:
        return ''

def accuracy_score(prediction, ground_truths):
    return exact_match_score(extract_answer(prediction), ground_truths)

def extract_support_ids(prediction):
    pattern = r"<relevance>\s*(.*?)\s*</relevance>"
    match = re.search(pattern, prediction, re.DOTALL | re.MULTILINE)
    if match:
        return match.group(1)
    else:
        return ''

def relevance_score(prediction, ground_truths_list):
    support_ids = extract_support_ids(prediction)
    try:
        support_ids_list = json.loads(support_ids)
        support_ids_list = sorted(list(set(support_ids_list)))
        if support_ids_list == ground_truths_list:
            return 1.0
        elif len(set(support_ids_list) & set(ground_truths_list)) > 0:
            return 0.5
        else:
            return 0.0
    except:
        return 0.0

def get_all_scores(prediction, answers_list, support_ids_list):
    em_score_now = exact_match_score(extract_answer(prediction), answers_list)
    f1_score_now = f1_score(extract_answer(prediction), answers_list)

    format_score_now = format_score(prediction)
    accuracy_score_now = accuracy_score(prediction, answers_list)
    relevance_score_now = relevance_score(prediction, support_ids_list)
    
    bonus_score = 10.0 if format_score_now == 1.0 and accuracy_score_now == 1.0 and relevance_score_now == 1.0 else 0.0

    return {
        'em_score': em_score_now,
        'f1_score': f1_score_now,
        'format_score': format_score_now,
        'accuracy_score': accuracy_score_now,
        'relevance_score': relevance_score_now,
        'bonus_score': bonus_score
    }

def eval_jsonl_rulebase(input_file, output_file):
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    results = []

    total = 0
    em_all = 0
    f1_all = 0
    format_all = 0
    accuracy_all = 0
    relevance_all = 0
    bonus_all = 0

    for item in data:
        prediction = item['model_response']
        if prediction is None:
            continue
        ground_truths = item['answers']
        try:
            support_ids = item['supporting_ids']
        except:
            support_ids = []


        all_scores = get_all_scores(prediction, ground_truths, support_ids)
        
        results.append({
            'prompt': item['prompt'],
            'model_response': prediction,
            'answers': ground_truths,
            'supporting_ids': support_ids,
            'em_score': all_scores['em_score'],
            'f1_score': all_scores['f1_score'],
            'format_score': all_scores['format_score'],
            'accuracy_score': all_scores['accuracy_score'],
            'relevance_score': all_scores['relevance_score'],
            'bonus_score': all_scores['bonus_score'],
        })
        em_all += all_scores['em_score']
        f1_all += all_scores['f1_score']
        format_all += all_scores['format_score']
        accuracy_all += all_scores['accuracy_score']
        relevance_all += all_scores['relevance_score']
        bonus_all += all_scores['bonus_score']

        total += 1
    
    em_all /= total
    f1_all /= total
    format_all /= total
    accuracy_all /= total
    relevance_all /= total
    bonus_all /= total

    print(f'eval_jsonl: {input_file} -> {output_file}')
    print(f'total: {total}, em_score: {em_all}, f1_score: {f1_all}, format_score: {format_all}, accuracy_score: {accuracy_all}, relevance_score: {relevance_all}, bonus_score: {bonus_all}')
    
    output_data = {
        'total': total,
        'em_score': em_all,
        'f1_score': f1_all,
        'format_score': format_all,
        'accuracy_score': accuracy_all,
        'relevance_score': relevance_all,
        'bonus_score': bonus_all,
        'details': results,
    }

    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='hotpotqa', help='Dataset name')
    parser.add_argument('--result-name', type=str, default='test', help='Result file name (without extension)')
    args = parser.parse_args()

    name = args.name
    result_name = args.result_name

    input_file = f'../results/vllm_inference_results/{name}/{result_name}.jsonl'

    output_dir = f'../results/eval_rulebase/{name}'
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f'{result_name}.json')

    print(f'Input file: {input_file}')
    print(f'Output file: {output_file}')

    eval_jsonl_rulebase(input_file, output_file)
