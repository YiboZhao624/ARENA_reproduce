import re
import json
import string

from tqdm import tqdm
import os
import time

# from gpt_api import openai_call

from evaluate import format_score, accuracy_score, relevance_score, get_all_scores, extract_answer

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

def format_prompt(model_response, correct_answer):
    prompt = '*************Consider a knowledge Q&A RAG task to test the capability of a testing model, the correct answer list is:*************\n' + correct_answer
    prompt += '\n\n\n\n*************Here is the model\'s response:*************\n' + model_response
    prompt += '\n\n\n\n*************Please check if the model\'s answer is correct. As long as the model\'s answer hits any item (or synonym) in the correct answer list, it can be considered correct. You only need to answer "yes" or "no".*************'
    return prompt

def get_score(response):
    response = response.lower()
    
    if response.startswith('yes'):
        return 1.0
    if response.startswith('no'):
        return 0.0

    if 'yes' in response:
        return 1.0
    elif 'no' in response:
        return 0.0
    else:
        return 0.5


def format_reward(completions, **kwargs):
    try:
        completion_contents = [completion[0]["content"] for completion in completions]
    except:
        completion_contents = completions
    
    scores = []
    for content in completion_contents:
        scores.append(format_score(content))
    return scores

def accuracy_reward(completions, answers, **kwargs):
    try:
        completion_contents = [completion[0]["content"] for completion in completions]
    except:
        completion_contents = completions

    # for llm-as-a-judge
    # prompts = []
    # for content, answer in zip(completion_contents, answers):
    #     try:
    #         prompts.append(format_prompt(extract_answer(content), str(answer)))
    #     except:
    #         prompts.append('')
    # scores = []
    # for prompt in prompts:
    #     if prompt == '':
    #         scores.append(0.0)
    #     else:
    #         response = openai_call(prompt)
    #         scores.append(get_score(response))
    
    scores = []
    for content, answer in zip(completion_contents, answers):
        scores.append(accuracy_score(content, answer))
    
    # print(f'len(scores): {len(scores)}, len(completions): {len(completions)}')
    assert len(scores) == len(completions)
    
    return scores

def relevance_reward(completions, supporting_ids, **kwargs):
    try:
        completion_contents = [completion[0]["content"] for completion in completions]
    except:
        completion_contents = completions
    
    scores = []
    for content, support_ids in zip(completion_contents, supporting_ids):
        scores.append(relevance_score(content, support_ids))
    return scores

def bonus_reward(completions, answers, supporting_ids, **kwargs):
    try:
        completion_contents = [completion[0]["content"] for completion in completions]
    except:
        completion_contents = completions
    
    scores = []
    for content, answer, support_ids in zip(completion_contents, answers, supporting_ids):
        scores.append(get_all_scores(content, answer, support_ids)['bonus_score'])
    return scores
