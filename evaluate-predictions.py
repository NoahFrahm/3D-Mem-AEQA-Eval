# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
DISK = "/playpen-nas-ssd4/nofrahm/Embodied/3D-Mem-AEQA-Eval/model_downloads"
os.environ["HF_HOME"] = DISK # hub + token
os.environ["HUGGINGFACE_HUB_CACHE"] = f"{DISK}/hub" # repos
os.environ["TRANSFORMERS_CACHE"] = f"{DISK}/transformers" # model weights
os.environ["HF_DATASETS_CACHE"] = f"{DISK}/datasets" # Arrow datasets
os.environ["TORCH_HOME"] = f"{DISK}/torch" # torch-hub models

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from openeqa.evaluation.llm_match import get_llm_match_score, QwenLoader, get_Qwen_llm_match_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results",
        type=Path,
        help="path to a results file",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default="data/open-eqa-v0.json",
        help="path to dataset (default: data/open-eqa-v0.json)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default="data/metrics",
        help="path to an output directory (default: data/metrics)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="evaluate results even if responses are missing (default: false)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print verbose outputs (default: false)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only evaluate the first 5 questions",
    )
    args = parser.parse_args()
    assert args.results.exists()
    assert args.dataset.exists()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.output_path = args.output_directory / (args.results.stem + "-metrics.json")
    if args.verbose:
        print("output path: {}".format(args.output_path))
    return args


def main(args: argparse.Namespace):

    # load LLM (Llama3?)
    _ = QwenLoader()

    # load results
    results = json.load(args.results.open("r"))
    # remove item if anser is null
    results = [item for item in results if item["answer"]]
    results_question_ids = [item["question_id"] for item in results]
    question_id_to_result = {result["question_id"]: result for result in results}
    print("found {:,} results".format(len(results)))

    # load dataset
    dataset = json.load(args.dataset.open("r"))
    dataset_question_ids = [item["question_id"] for item in dataset]
    question_id_to_item = {item["question_id"]: item for item in dataset}
    # keep only existing questions
    dataset_question_ids = [quest_id for quest_id in dataset_question_ids if quest_id in results_question_ids]
    dataset = [item for item in dataset if item["question_id"] in dataset_question_ids]
    question_id_to_item = {item["question_id"]: item for item in dataset}
    print("found {:,} questions".format(len(dataset)))

    # check that results and dataset match
    if not args.force:
        assert len(dataset_question_ids) == len(results_question_ids)
        assert set(dataset_question_ids) == set(results_question_ids)

    # load scores
    all_scores = {}
    if args.output_path.exists():
        all_scores = json.load(args.output_path.open("r"))
        print(f"found {len(all_scores)} existing scores in {args.output_path}")

    # evaluate predictions
    for idx, question_id in enumerate(tqdm(results_question_ids)):
        if args.dry_run and idx >= 5:
            break

        if question_id in all_scores:
            continue

        item = question_id_to_item[question_id]
        result = question_id_to_result[question_id]
        extra_answers = item["extra_answers"] if "extra_answers" in item else None

        # pre-process answers
        if result["answer"]:
            # remove anything after the last period
            end_idx = result["answer"].rfind(".")
            if end_idx >= 0 and end_idx + 1 < len(result["answer"]):
                result["answer"] = result["answer"][: end_idx + 1]

        # score = get_llm_match_score(
        score = get_Qwen_llm_match_score(
            question=item["question"],
            answer=item["answer"],
            prediction=result["answer"],
            extra_answers=extra_answers,
        )

        all_scores[question_id] = score
        json.dump(all_scores, args.output_path.open("w"), indent=2)

    # calculate final score
    scores = np.array(list(all_scores.values()))
    scores = 100.0 * (np.clip(scores, 1, 5) - 1) / 4
    print("final score: {:.1f}".format(np.mean(scores)))


if __name__ == "__main__":
    main(parse_args())
