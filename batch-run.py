import subprocess
import os

dir = os.path.dirname(os.path.abspath(__file__))

folders = [
    "/playpen-nas-ssd4/nofrahm/Embodied/Multi-Mem/results-past_runs/exp_eval_aeqa_184-7B-baseline",
    "/playpen-nas-ssd4/nofrahm/Embodied/Multi-Mem/results-past_runs/exp_eval_aeqa_184-32B-baseline",
    "/playpen-nas-ssd4/nofrahm/Embodied/Multi-Mem/results-past_runs/exp_eval_aeqa_closest_frontier_baseline_184",
    "/playpen-nas-ssd4/nofrahm/Embodied/Multi-Mem/results-past_runs/exp_eval_aeqa_184-CDF-32B",
    "/playpen-nas-ssd4/nofrahm/Embodied/Multi-Mem/results-past_runs/exp_eval_aeqa_dynamic_swap_everystep_32B_184_run_bug_fixed",
    "/playpen-nas-ssd4/nofrahm/Embodied/Multi-Mem/results-past_runs/exp_eval_aeqa_dynamic_swap_everystep_184_run_bug_fixed",
]

server_endpoint_link = "http://zidane.cs.unc.edu:12182/Qwen_VL/infer" #Only use Qwen3 30B

with open(os.path.join(dir, 'data', 'log.txt'), 'r+') as file:
    file.truncate(0)

for exp in folders:
    filename = "gpt_answer-metrics-" + os.path.basename(exp).replace('exp_eval_aeqa_', '') + ".json"
    if os.path.exists(os.path.join(dir, 'data', 'metrics', filename)):
        os.remove(os.path.join(dir, 'data', 'metrics', filename))
    if os.path.exists(os.path.join(dir, 'data', 'metrics', "full_"+filename)):
        os.remove(os.path.join(dir, 'data', 'metrics', "full_"+filename))

    status, output = subprocess.getstatusoutput(f"python evaluate-expressbench.py --dataset /playpen-nas-ssd3/prakrut/Multi-Mem/data/aeqa_questions-184.json --filename {filename} --server_endpoint_link {server_endpoint_link} {exp}/gpt_answer.json")
    if status != 0:
        print(output)
    else:
        status, output = subprocess.getstatusoutput(f"python get-scores-grounded.py --dataset open-eqa-184 --filename {filename} --result-path {exp}")