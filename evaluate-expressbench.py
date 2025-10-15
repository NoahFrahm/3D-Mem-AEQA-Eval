import os
import argparse
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import requests
from PIL import Image
import io, base64
import pickle

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
    parser.add_argument(
        "--filename",
        type=str,
    )

    parser.add_argument(
        "--server_endpoint_link",
        type=str,
    )
    
    args = parser.parse_args()
    assert args.results.exists()
    assert args.dataset.exists()
    
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.output_path = args.output_directory / (args.filename)
    args.output_full = args.output_directory / ("full_"+args.filename)
    if args.verbose:
        print("output path: {}".format(args.output_path))
    return args

def qwen_call(server_endpoint_link, content):
    resp = requests.post(
        server_endpoint_link,
        json={"prompt": content},
        timeout=300
    )
    return resp

def resize_image(image, target_h, target_w):
    image = Image.fromarray(image)
    image = image.resize((target_w, target_h))
    return np.array(image)

def image_to_base64(path):
    img = Image.open(path).convert("RGB").resize((360, 360))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")
    

def build_chat_prompt(question: str, answer: str, response: str, image):
    system_msg = {
        "role": "system",
        "content": "You are an AI assistant who will help me to evaluate the response given the question, the correct answer and the scene observed by the robot."
    }

    user_text = f"""The input includes the Question, the Answer, the Response given by the model and the Image of the environment. You need to evaluate the alignment between the Response and the Image, as well as between the Response and the Answer, and assign a score for each.
First, assess whether the Response depends on the observed environment Image and assign one of three possible scores [0, 0.5, 1]. If the target object referenced in the Question or the Answer is present in the Image and is described accurately, assign a score of 1. If the object is present but inaccurately described, assign a score of 0.5. If the object does not exist in the Image, meaning the answer is entirely unrelated to the Image and fabricated, assign a score of 0.
Additionally, compare the model's Response with the Answer and Image, assigning a score scale from 1 to 5 based on its accuracy.

Your output should consist of exactly two fractions, separated by a comma. No further elaboration is necessary. Please provide the output that fulfills these criteria given the input.

Question: {question}
Answer: {answer}
Response: {response}
"""

    # Proper multimodal format
    user_msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": user_text},
            {"type": "image", "image": image},
        ],
    }

    return [system_msg, user_msg]

def main(args: argparse.Namespace):
    results = json.load(args.results.open("r"))
    results = [item for item in results if item["answer"]]
    results_question_ids = [item["question_id"] for item in results]
    question_id_to_result = {result["question_id"]: result for result in results}
    print("found {:,} results".format(len(results)))
    
    
    dataset = json.load(args.dataset.open("r"))
    dataset_question_ids = [item["question_id"] for item in dataset]
    question_id_to_item = {item["question_id"]: item for item in dataset}
    
    dataset_question_ids = [quest_id for quest_id in dataset_question_ids if quest_id in results_question_ids]
    dataset = [item for item in dataset if item["question_id"] in dataset_question_ids]
    question_id_to_item = {item["question_id"]: item for item in dataset}
    print("found {:,} questions".format(len(dataset)))
    
    question_step_info = json.load(Path(os.path.join(os.path.dirname(args.results), 'question_step_info_0.0_1.0.json')).open("r"))
    
    with open(os.path.join(os.path.dirname(args.results), 'path_length_list.pkl'), 'rb') as file:
        path_length = pickle.load(file)
    all_scores = {}
    all_scores_with_length = {}
    if args.output_path.exists():
        all_scores = json.load(args.output_path.open("r"))
        print(f"found {len(all_scores)} existing scores in {args.output_path}")
        
    
    for idx, question_id in enumerate(tqdm(results_question_ids)):
        if args.dry_run and idx >= 5:
            break

        if question_id in all_scores:
            continue
        all_scores[question_id] = {"confidence_snapshot": {}, "chosen_snapshot": {}}
        item = question_id_to_item[question_id]
        question=item["question"]
        answer=item["answer"]
        category=item["category"]
        result = question_id_to_result[question_id]

        # pre-process answers
        if result["answer"]:
            # remove anything after the last period
            end_idx = result["answer"].rfind(".")
            if end_idx >= 0 and end_idx + 1 < len(result["answer"]):
                result["answer"] = result["answer"][: end_idx + 1]
        prediction=result["answer"]
        
        qsi_items = question_step_info[question_id].keys()
        qsi_items = [int(x) for x in qsi_items]
        qsi_step = question_step_info[question_id][str(max(qsi_items))]
        snapshots = qsi_step['rgb_memory_snapshot_paths']
        confidence_values = qsi_step['confidence_values'][:len(snapshots)]
        confident_snapshot = snapshots[confidence_values.index(max(confidence_values))]
        
        image = image_to_base64(os.path.join(os.path.dirname(args.results), question_id, 'snapshot', confident_snapshot))
        image = f"data:image/jpeg;base64,{image}"

        content = build_chat_prompt(question=question, answer=answer, response=prediction, image=image)
        
        resp = qwen_call(server_endpoint_link=args.server_endpoint_link, content=content)
        try:
            grounded, match = resp.json()['text'].split(',')
            grounded = str(grounded).strip()
            match = str(match).strip()
            assert grounded in {"0", "0.5", "1"}, f"Invalid value: {grounded}"
            assert match in {"1", "2", "3", "4", "5"}, f"Invalid value: {match}"
            score = (float(grounded) * float(match))
        except:
            raise
        
        do_snaphot = True
        snapshot_dir = os.path.join(os.path.dirname(args.results), question_id, 'chosen_snapshot')
        chosen_snapshot = None
        if os.path.isdir(snapshot_dir) and do_snaphot:
            chosen_snapshot = os.listdir(snapshot_dir)
            if chosen_snapshot:
                if chosen_snapshot[0].replace('snapshot_', '') != confident_snapshot:
                            image = image_to_base64(os.path.join(snapshot_dir, chosen_snapshot[0]))
                            image = f"data:image/jpeg;base64,{image}"

                            content = build_chat_prompt(question=question, answer=answer, response=prediction, image=image)

                            resp = qwen_call(server_endpoint_link=args.server_endpoint_link, content=content)

                            try:
                                grounded, match = resp.json()['text'].split(',')
                                grounded = str(grounded).strip()
                                match = str(match).strip()
                                assert grounded in {"0", "0.5", "1"}, f"Invalid value: {grounded}"
                                assert match in {"1", "2", "3", "4", "5"}, f"Invalid value: {match}"
                                score = (float(grounded) * float(match))
                            except:
                                raise
        
        all_scores[question_id] = score
        all_scores_with_length[question_id] = {
            "score": score,
            "grounded": grounded,
            "accuracy": match,
            "path": path_length.get(question_id),
            "category": category
        }
        json.dump(all_scores, args.output_path.open("w"), indent=2)
        json.dump(all_scores_with_length, args.output_full.open("w"), indent=2)

    # calculate final score
    scores = np.array(list(all_scores.values()))
    scores = np.mean(100.0 * (np.clip(scores, 0, 5) / 5))
    print("final score: {:.1f}".format(np.mean(scores)))
    
    
if __name__ == "__main__":
    main(parse_args())