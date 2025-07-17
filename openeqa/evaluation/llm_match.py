# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import traceback
from typing import Optional

from openeqa.utils.openai_utils import (
    call_openai_api,
    prepare_openai_messages,
    set_openai_key,
)
from openeqa.utils.prompt_utils import load_prompt

import torch
from transformers import pipeline
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import Optional



def parse_score(output: str, tag: str = "Your mark:") -> str:
    if output.isdigit():
        return int(output)
    start_idx = output.find(tag)
    if start_idx == -1:
        raise ValueError("Invalid output string: {}".format(output))
    end_idx = output.find("\n", start_idx)
    if end_idx == -1:
        return int(output[start_idx:].replace(tag, "").strip())
    return int(output[start_idx:end_idx].replace(tag, "").strip())


def get_llm_match_score(
    question: str,
    answer: str,
    prediction: str,
    extra_answers: Optional[list] = None,
    openai_key: Optional[str] = None,
    openai_model: str = "gpt-4-1106-preview",
    openai_seed: int = 1234,
    openai_max_tokens: int = 32,
    openai_temperature: float = 0.2,
    verbose: bool = False,
):
    if prediction is None:
        return 0

    prompt_name = "mmbench" if extra_answers is None else "mmbench-extra"
    prompt = load_prompt(prompt_name)

    try:
        set_openai_key(key=openai_key)
        messages = prepare_openai_messages(
            prompt.format(
                question=question,
                answer=answer,
                prediction=prediction,
                extra_answers=extra_answers,
            ),
        )
        output = call_openai_api(
            messages=messages,
            model=openai_model,
            seed=openai_seed,
            max_tokens=openai_max_tokens,
            temperature=openai_temperature,
            verbose=verbose,
        )
        return parse_score(output)
    except Exception as e:
        traceback.print_exc()
        raise e


# custom llm match score method using qwen instead of OpenAI
def get_Qwen_llm_match_score(
    question: str,
    answer: str,
    prediction: str,
    extra_answers: Optional[list] = None,
    openai_key: Optional[str] = None,
    openai_model: str = "gpt-4-1106-preview",
    openai_seed: int = 1234,
    openai_max_tokens: int = 32,
    openai_temperature: float = 0.2,
    verbose: bool = False,
):
    if prediction is None:
        return 0

    prompt_name = "mmbench" if extra_answers is None else "mmbench-extra"
    prompt = load_prompt(prompt_name)

    messages = prepare_openai_messages(
        prompt.format(
            question=question,
            answer=answer,
            prediction=prediction,
            extra_answers=extra_answers,
        ),
    )

    model = QwenLoader()

    prompt_answer = model.infer(
        prompt=messages,
        temperature=0.7,
        top_p=0.95,
        max_new_tokens=4096,  # Adjusted to match the original intent
    )

    score = parse_score(prompt_answer[0]) 
    breakpoint()
    
    return score

    # call_qwen_api(
    #     sys_prompt,
    #     contents,
    #     messages=messages,
    #     seed=seed,
    #     max_tokens=max_tokens,
    #     temperature=temperature,
    # )

    # output = call_openai_api(
    #     messages=messages,
    #     model=openai_model,
    #     seed=openai_seed,
    #     max_tokens=openai_max_tokens,
    #     temperature=openai_temperature,
    #     verbose=verbose,
    # )
    
    # return parse_score(output) 


# class LlamaLoader():
#     """
#     A Singleton class to load and manage models. This ensures models are loaded
#     only once and provides a global access point.
#     """
#     _instance = None
#     _models = {}
#     model_id = "meta-llama/Llama3.3-70B-Instruct" # <-- llama 3.3
#     model_name = "Llama3.3"

#     def __new__(cls):
#         """
#         This special method controls the creation of instances.
#         It ensures only one instance of ModelLoader is ever created.
#         """
#         if cls._instance is None:
#             print("INFO: Creating the first ModelLoader instance. Loading models...")
#             cls._instance = super(LlamaLoader, cls).__new__(cls)
#             cls._instance._load_model()
#         return cls._instance

#     def _load_model(self):
#         """
#         A private method to load Qwen model.
#         """
#         print("INFO: Loading 'Llama' model...")

#         device = "cuda" if torch.cuda.is_available() else "cpu"

#         generator = pipeline(model=self.model_id, device=device, torch_dtype=torch.bfloat16)

#         self._models[self.model_name] = generator

#         # Generation:
#         # [
#         #   {'role': 'system', 'content': 'You are a helpful assistant, that responds as a pirate.'},
#         #   {'role': 'user', 'content': "What's Deep Learning?"},
#         #   {'role': 'assistant', 'content': "Yer lookin' fer a treasure trove o'
#         #             knowledge on Deep Learnin', eh? Alright then, listen close and
#         #             I'll tell ye about it.\n\nDeep Learnin' be a type o' machine
#         #             learnin' that uses neural networks"}
#         # ]

#         print("INFO: Loaded 'Llama' model successfully.")

#     def infer(self, prompt: list, temperature: float = 0.7, top_p: float = 0.95, max_new_tokens: int = 4096) -> list:
#         """
#         Llama inference
#         """

#         # prompt = [
#         #     {"role": "system", "content": "You are a helpful assistant, that responds as a pirate."},
#         #     {"role": "user", "content": "What's Deep Learning?"},
#         # ]

#         generator = self._models.get(self.model_name)

#         generation = generator(
#             prompt,
#             do_sample=False,
#             temperature=1.0,
#             top_p=1,
#             max_new_tokens=50
#         )

#         print(f"Generation: {generation[0]['generated_text']}")

#         return generation[0]


class QwenLoader():
    """
    A Singleton class to load and manage models. This ensures models are loaded
    only once and provides a global access point.
    """
    _instance = None
    _models = {}
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct" # needs local storage due to large size
    # model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    model_name = "Qwen2_5_VL"

    def __new__(cls):
        """
        This special method controls the creation of instances.
        It ensures only one instance of ModelLoader is ever created.
        """
        if cls._instance is None:
            print("INFO: Creating the first ModelLoader instance. Loading models...")
            cls._instance = super(QwenLoader, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        """
        A private method to load Qwen model.
        """
        print("INFO: Loading 'Qwen2_5_VL' model...")

        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype="auto", device_map="auto"
        )
        
        processor = AutoProcessor.from_pretrained(
            self.model_id,
            force_download=True
        )

        self._models[self.model_name] = (processor, model)

        print("INFO: Loaded 'Qwen2_5_VL' model successfully.")

    def infer(self, prompt: list, temperature: float = 0.7, top_p: float = 0.95, max_new_tokens: int = 4096) -> list:
        """
        prompt: [
            {
                "role": "user",
                "content": [
                    {"type":"image","image":"file://..."},
                    {"type":"text","text":"Describe this image."}
                ]
            }
        ]
        """
        processor, model = self._models.get(self.model_name)

        text = processor.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device) # this is problematic for distributed inference

        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # Trim off prompt tokens before decoding
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
        
        return processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )


def qwen_format_content(contents):
    formated_content = []
    for c in contents:
        formated_content.append({"type": "text", "text": c[0]})
        if len(c) == 2:
            formated_content.append(
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{c[1]}"
                }
            )
    return formated_content


def call_qwen_api(sys_prompt, contents) -> Optional[str]:
    max_tries = 5
    retry_count = 0
    formated_content = qwen_format_content(contents)
    message_text = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": formated_content},
    ]

    qwen_model = QwenLoader()

    while retry_count < max_tries:
        try:
            prompt_answer = qwen_model.infer(
                prompt=message_text,
                temperature=0.7,
                top_p=0.95,
                max_new_tokens=4096,  # Adjusted to match the original intent
            )
            return prompt_answer[0]
        except Exception as e:
            print("Error: ", e)
            # time.sleep(60)
            retry_count += 1
            continue

    return None






if __name__ == "__main__":
    # example usage
    question = "What color is the rug?"
    answer = "tan with pink and blue"
    prediction = "brown with pink and blue"
    score = get_llm_match_score(question, answer, prediction)
    print("*" * 40)
    print("example question:    {}".format(question))
    print("ground-truth answer: {}".format(answer))
    print("predicted answer:    {}".format(prediction))
    print("llm-match score:     {}".format(score))
    print("*" * 40)
