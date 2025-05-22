from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoTokenizer
import torch
from typing import List, Union
import json
import os
import argparse

def apply_chat_template(
    example,
    tokenizer,
    tools = None,
) -> dict[str, str]:
    r"""
    Apply a chat template to a conversational example along with the schema for a list of functions in `tools`.

    For more details, see [`maybe_apply_chat_template`].
    """
    # Check that the example has the correct keys
    supported_keys = ["prompt", "chosen", "rejected", "completion", "messages", "label"]
    example_keys = {key for key in example.keys() if key in supported_keys}
    if example_keys not in [
        {"messages"},  # language modeling
        {"prompt"},  # prompt-only
        {"prompt", "completion"},  # prompt-completion
        {"prompt", "chosen", "rejected"},  # preference
        {"chosen", "rejected"},  # preference with implicit prompt
        {"prompt", "completion", "label"},  # unpaired preference
    ]:
        raise KeyError(f"Invalid keys in the example: {example_keys}")

    # Apply the chat template to the whole conversation
    if "messages" in example:
        messages = tokenizer.apply_chat_template(example["messages"], tools=tools, tokenize=False)

    # Apply the chat template to the prompt, adding the generation prompt
    if "prompt" in example:
        last_role = example["prompt"][-1]["role"]
        if last_role == "user":
            add_generation_prompt = True
            continue_final_message = False
        elif last_role == "assistant":
            add_generation_prompt = False
            continue_final_message = True
        else:
            raise ValueError(f"Invalid role in the last message: {last_role}")
        prompt = tokenizer.apply_chat_template(
            example["prompt"],
            tools=tools,
            continue_final_message=continue_final_message,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    # Apply the chat template to the entire prompt + completion
    if "prompt" in example:  # explicit prompt and prompt-completion case
        if "chosen" in example:
            prompt_chosen = tokenizer.apply_chat_template(
                example["prompt"] + example["chosen"], tools=tools, tokenize=False
            )
            chosen = prompt_chosen[len(prompt) :]
        if "rejected" in example and "prompt" in example:  # explicit prompt
            prompt_rejected = tokenizer.apply_chat_template(
                example["prompt"] + example["rejected"], tools=tools, tokenize=False
            )
            rejected = prompt_rejected[len(prompt) :]
        if "completion" in example:
            prompt_completion = tokenizer.apply_chat_template(
                example["prompt"] + example["completion"], tools=tools, tokenize=False
            )
            completion = prompt_completion[len(prompt) :]
    else:  # implicit prompt case
        if "chosen" in example:
            chosen = tokenizer.apply_chat_template(example["chosen"], tools=tools, tokenize=False)
        if "rejected" in example:
            rejected = tokenizer.apply_chat_template(example["rejected"], tools=tools, tokenize=False)

    # Ensure that the prompt is the initial part of the prompt-completion string
    if "prompt" in example:
        error_message = (
            "The chat template applied to the prompt + completion does not start with the chat template applied to "
            "the prompt alone. This can indicate that the chat template is not supported by TRL."
            "\n**Prompt**:\n{}\n\n**Prompt + Completion**:\n{}"
        )
        if "chosen" in example and not prompt_chosen.startswith(prompt):
            raise ValueError(error_message.format(prompt, prompt_chosen))
        if "rejected" in example and not prompt_rejected.startswith(prompt):
            raise ValueError(error_message.format(prompt, prompt_rejected))
        if "completion" in example and not prompt_completion.startswith(prompt):
            raise ValueError(error_message.format(prompt, prompt_completion))

    # Extract the completion by removing the prompt part from the prompt-completion string
    output = {}
    if "messages" in example:
        output["text"] = messages
    if "prompt" in example:
        output["prompt"] = prompt
    if "chosen" in example:
        output["chosen"] = chosen
    if "rejected" in example:
        output["rejected"] = rejected
    if "completion" in example:
        output["completion"] = completion
    if "label" in example:
        output["label"] = example["label"]

    return output

class VLLMWrapper:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
        max_model_len: int = None,
        temperature: float = 0.9,
        max_completion_length: int = 1024,
        guided_decoding_regex: str = None,
        num_generations: int = 1,
        processing_class=None,
    ):
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.processing_class = processing_class

        if self.processing_class.pad_token is None:
            self.processing_class.pad_token = self.processing_class.eos_token
        if self.processing_class.pad_token_id is None:
            self.processing_class.pad_token_id = self.processing_class.eos_token_id

        self.llm = LLM(
            model=model_name,
            device=device,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            enable_prefix_caching=True,
            max_model_len=max_model_len,
        )

        guided_decoding = None
        if guided_decoding_regex is not None:
            guided_decoding = GuidedDecodingParams(backend="outlines", regex=guided_decoding_regex)

        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_completion_length,
            guided_decoding=guided_decoding,
            n=num_generations,
        )

    def generate(self, inputs: List[Union[str, List[dict]]]):
        device = "cuda:0"
        prompts_text = [apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        outputs = self.llm.generate(prompts_text, self.sampling_params)

        completion_ids = []
        for output in outputs:
            for seq in output.outputs:
                completion_ids.append(seq.token_ids)
        
        completion_tensors = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = torch.nn.utils.rnn.pad_sequence(
            completion_tensors,
            batch_first=True,
            padding_value=self.processing_class.pad_token_id
        )

        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        return completions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='hotpotqa', help='Name of the dataset')
    parser.add_argument('--testdata-name', type=str, default='test', help='Input JSONL test data file name')
    parser.add_argument('--saved-name', type=str, default='test', help='Output result file name')
    parser.add_argument('--model-path', type=str, default='../model/Qwen2.5-7B-Instruct', help='Path to the model')
    args = parser.parse_args()

    name = args.name
    testdata_name = args.testdata_name
    saved_name = args.saved_name
    model_path = args.model_path

    # Initialize the model wrapper
    wrapper = VLLMWrapper(
        model_name=model_path,
        temperature=0.9,
        max_completion_length=1024,
        num_generations=1
    )

    # Load input data
    testdata_path = f'../data/data_conversation/{name}/{testdata_name}.jsonl'
    data = []
    with open(testdata_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    # Generate model responses
    completions = wrapper.generate(inputs=data)

    # Clean up NCCL resources if using torch.distributed
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    # Combine prompts and responses into results
    results = []
    for i, item in enumerate(data):
        results.append({
            'prompt': item['prompt'],
            'model_response': completions[i],
            'answers': item['answers'],
            'supporting_ids': item.get('supporting_ids', []),
            'name': item['name'],
        })

    # Save the generated results
    output_dir = f'../results/vllm_inference_results/{name}'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'{saved_name}.jsonl')
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
