This repo is for A-EQA evaluation

### Installation

The code requires a `python>=3.9` environment. We recommend using conda:

```bash
conda create -n openeqa python=3.9
conda activate openeqa
pip install -r requirements.txt
pip install -e .
```

### Running evaluations

Automatic evaluation is implemented with GPT-4 using the prompts found [here](prompts/mmbench.txt) and [here](prompts/mmbench-extra.txt).

```bash
# set the OPENAI_API_KEY environment variable to your personal API key
python evaluate-predictions.py --dataset data/<dataset>.json <path/to/results>/gpt_answer.json

python get-scores.py --dataset <dataset> --result-path <path/to/results>
```
