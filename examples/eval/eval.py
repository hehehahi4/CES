import os
import re
import json
from collections import deque
from PIL import Image
from tqdm import tqdm
import torch
from multiprocessing import Process, Queue, set_start_method
from typing import List, Dict, Tuple, Any
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


PLANNER_PROMPT = """You are a GUI task coordinator Agent. Your role is to actively collaborate with the Executor Agent to complete complex GUI navigation tasks. Given a high-level task description and the current state of the task, your goal is to provide a clear and precise fine-grained instruction for the Executor Agent to help accomplish the task.
Screenshot: <image>
High-level task: {high_level_instruction}
Current_state: {current_state}
First, think step-by-step. Put your reasoning within <think> tags.
After your reasoning, provide the instruction within <answer> tags."""


MEMORY_PROMPT_stc = """You are a GUI task State Tracker Agent. Your core function is dynamic context compression and state updating. You will receive the high-level user instruction, the previous task state (a summary of progress up to the last step), and the latest output of executor agent. Your task is to generate the new task state. This should be a high-semantic natural language summary that updates the previous state based on the latest action, maintaining a coherent record of the task's progress.
High-level user instruction: {high_level_instruction}
Latest output of executor agent: {current_action}
Previous Task State: {current_state}
"""

# TODO: Change it by different models and different benchmark's action space
ACTION_AGENT_PROMPT = """You are GUI executor Agent, a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue executing the command '{instruction}'.
Please provide the action to perform (enumerate from ['press_back', 'press_recent', 'press_home', 'complete', 'long_press', 'scroll', 'click', 'impossible', 'type']), the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.
Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:
<think> ... </think> <answer>[{{'action': enum['click', 'type', 'scroll', 'long_press', 'press_back', 'press_recent', 'press_home', 'complete'], 'point': [x, y], 'input_text': 'no input text [default]'}}]</answer>
Note: 
specific input text (no default) is necessary for actions 'type' and 'scroll'
specific point is necessary for actions 'click' and 'long_press'
Example answer: 
[{{'action': 'click', 'point': [123, 300], 'input_text': 'no input text'}}]
[{{'action': 'type', 'point': [-100, -100], 'input_text': 'shanghai shopping mall'}}]
[{{'action': 'scroll', 'point': [-100, -100], 'input_text': enum['up', 'left', 'right', 'down']}}]
[{{'action': enum['complete','press_back', 'press_recent', 'press_home'], 'point': [-100, -100], 'input_text': 'no input text'}}]
"""


class MemoryCache:
    def __init__(self, stc_capacity=4):
        self.stc_queue = deque(maxlen=stc_capacity)
    def update_stc(self, summary: str):
        self.stc_queue.append(summary)
    def get_stc_for_prompt(self) -> str:
        if not self.stc_queue:
            return "None. This is the first step."
        return "\n".join(f"- {s}" for s in self.stc_queue)
    def reset(self):
        self.stc_queue.clear()

def extract_from_tags(text: str, tag: str) -> str:
    match = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    print(f"Could not find <{tag}> tag in:\n{text}")
    if tag == "answer":
        return "COMPLETE"
    return f"Error: No <{tag}> tag found."

def text_agent_worker(model_path: str, device_str: str, input_queue: Queue, output_queue: Queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = device_str.split(":")[1]
    COMMON_VLLM_KWARGS = dict(dtype="bfloat16", tensor_parallel_size=1, gpu_memory_utilization=0.9, max_model_len=8192, trust_remote_code=True, load_format="safetensors")

    llm = LLM(model=model_path, **COMMON_VLLM_KWARGS)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=512)

    while True:
        request_id, prompt_text = input_queue.get()
        if prompt_text is None: break
        
        messages = [{"role": "user", "content": prompt_text}]
        final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate([final_prompt], sampling_params)
        output_queue.put((request_id, outputs[0].outputs[0].text))


def multimodal_agent_worker(model_path: str, device_str: str, input_queue: Queue, output_queue: Queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = device_str.split(":")[1]
    COMMON_VLLM_KWARGS = dict(dtype="bfloat16", tensor_parallel_size=1, gpu_memory_utilization=0.9, max_model_len=8192, trust_remote_code=True, enforce_eager=True)
    llm = LLM(model=model_path, **COMMON_VLLM_KWARGS)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=512)

    while True:
        request_id, prompt_text, image_path = input_queue.get()
        if prompt_text is None: break
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image"}]}]
        final_prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image = Image.open(image_path).convert("RGB")
        llm_inputs = {"prompt": final_prompt_text, "multi_modal_data": {"image": image}}
        outputs = llm.generate([llm_inputs], sampling_params)
        output_queue.put((request_id, outputs[0].outputs[0].text))



def parse_action_agent_output(model_output: str) -> Dict[str, Any]:
    result = {
        "pred_action": "impossible",
        "pred_coord": [-1, -1],
        "pred_input_text": "no input text"
    }
    
    answer_match = re.search(r"<answer>(.*?)</answer>", model_output, re.DOTALL)
        
    answer_content = answer_match.group(1).strip()
    actions_list = eval(answer_content)
    if isinstance(actions_list, list) and actions_list:
        action_dict = actions_list[0]
        if isinstance(action_dict, dict):
            result["pred_action"] = action_dict.get("action", "impossible").lower()
            result["pred_coord"] = action_dict.get("point", [-1, -1])
            result["pred_input_text"] = action_dict.get("input_text", "no input text")
    return result

def format_ground_truth(step_data: Dict[str, Any]) -> Dict[str, Any]:
    action_type = step_data['action']
    gt_action = "impossible"
    gt_bbox = [-1, -1]
    gt_input_text = "no input text"

    if action_type in ['CLICK', 'LONG_PRESS']:
        gt_action = action_type.lower()
        match = re.search(r'\((\d+),\s*(\d+)\)', step_data.get('ps', ''))
        if match:
            gt_bbox = [int(match.group(1)), int(match.group(2))]
    elif action_type == 'TEXT':
        gt_action = "type"
        gt_input_text = step_data.get('info', '')
    elif action_type == ('SCROLL'):
        gt_action = "scroll"
        ls = eval(step_data.get('ps', ''))
        print(ls)
        x1, y1, x2, y2 = ls[0][0], ls[0][1], ls[-1][0], ls[-1][1]
        if abs(x1-x2) > abs(y1-y2):
            if x1 > x2:
                gt_input_text = "right"
            else:
                gt_input_text = "left"
        else:
            if y1 > y2:
                gt_input_text = "up"
            else:
                gt_input_text = "down"
    elif action_type in ['PRESS_BACK', 'PRESS_HOME', 'COMPLETE']:
        gt_action = action_type.lower()
        
    return {
        "gt_action": gt_action,
        "gt_bbox": gt_bbox,
        "gt_input_text": gt_input_text
    }

def evaluate_trajectory_multigpu(episodes_to_evaluate: list,
                                 planner_q_in, planner_q_out,
                                 action_q_in, action_q_out,
                                 memory_q_in, memory_q_out):
    memory_cache = MemoryCache()
    all_final_results = []

    processor = AutoProcessor.from_pretrained(ACTION_MODEL_PATH, trust_remote_code=True)

    pbar = tqdm(episodes_to_evaluate, desc="Evaluating Trajectories")
    for episode_id in pbar:
        try:
            with open(f"/data1/datasets/GUIOdyssey/annotations/{episode_id}.json", 'r', encoding='utf-8') as f:
                data = json.load(f)

            high_level_instruction = data['task_info']['instruction']
            group = 'web'
            device_info = data.get('device_info', {}); w, h = device_info.get('w', 1000), device_info.get('h', 1000)
            memory_cache.reset()

            for step in data['steps']:
                step_num = step['step']
                screenshot_path = f"/data1/datasets/GUIOdyssey/screenshots/{episode_id}_{step_num}.png"
                if not os.path.exists(screenshot_path): continue

                request_id = f"{episode_id}_{step_num}"
                
                planner_prompt = PLANNER_PROMPT.format(high_level_instruction=high_level_instruction, current_state=memory_cache.get_stc_for_prompt())
                planner_q_in.put((request_id + "_plan", planner_prompt, screenshot_path))
                _, planner_output_raw = planner_q_out.get()
                print(planner_output_raw)
                lli = extract_from_tags(planner_output_raw, "answer")

                action_prompt = ACTION_AGENT_PROMPT.format(instruction=lli)
                action_q_in.put((request_id + "_act", action_prompt, screenshot_path))
                _, action_output_raw = action_q_out.get()

                print(memory_cache.get_stc_for_prompt())
                stc_prompt = MEMORY_PROMPT_stc.format(high_level_instruction=high_level_instruction, current_state=memory_cache.get_stc_for_prompt(), current_action=action_output_raw)
                memory_q_in.put((request_id + "_stc", stc_prompt))
                _, memory_output_raw = memory_q_out.get()
                print(memory_output_raw)
                memory_cache.update_stc(memory_output_raw)
                
                parsed_prediction = parse_action_agent_output(action_output_raw)

                pred_coord_raw = parsed_prediction["pred_coord"]
                
                original_image = Image.open(screenshot_path).convert("RGB")
                original_w, original_h = original_image.size
                print("original_w, original_h:", original_w, original_h)
                print(original_w==w, original_h==h)

                gt_scale_w = original_w / 1000
                gt_scale_h = original_h / 1000

                pred_coord_scaled = [
                    pred_coord_raw[0] / gt_scale_w,
                    pred_coord_raw[1] / gt_scale_h
                ]
                print(parsed_prediction["pred_coord"])
                print(pred_coord_scaled)
                parsed_prediction["pred_coord"] = pred_coord_scaled

                formatted_gt = format_ground_truth(step)
    
                final_result = {
                    "gt_action": formatted_gt["gt_action"],
                    "gt_bbox": formatted_gt["gt_bbox"],
                    "gt_input_text": formatted_gt["gt_input_text"],
                    "image_size": [w, h],
                    "pred_action": parsed_prediction["pred_action"],
                    "pred_coord": parsed_prediction["pred_coord"],
                    "pred_input_text": parsed_prediction["pred_input_text"],
                    "episode_id": episode_id,
                    "step": step_num,
                }
                
                all_final_results.append(final_result)


        except Exception as e:
            import traceback
            print(f"Main loop error on episode {episode_id}: {e}\n{traceback.format_exc()}")
            continue

    return all_final_results


if __name__ == "__main__":
    set_start_method("spawn", force=True)

    # TODO: change
    PLANNER_MODEL_PATH = "/data3/home/dengzehao/data/model/my_output/rl/planner-vl2"
    MEMORY_MODEL_PATH = "/data3/home/dengzehao/data/model/my_output/rl/tracker-4B"
    ACTION_MODEL_PATH = "/data3/home/dengzehao/data/model/GUI-R1-7B"
    ANNOTATIONS_DIR = "/data1/datasets/GUIOdyssey/annotations"
    RANDOM_SPLIT_FILE = "/data1/datasets/GUIOdyssey/splits/random_split.json"
    OUTPUT_DIR = "/data3/home/dengzehao/data/my/eval/output_json/ces/ody" 
    output_filename = f"guir17b2.jsonl"


    planner_input_q, planner_output_q = Queue(), Queue()
    action_input_q, action_output_q = Queue(), Queue()
    memory_input_q, memory_output_q = Queue(), Queue()
    planner_process = Process(target=multimodal_agent_worker, args=(PLANNER_MODEL_PATH, "cuda:4", planner_input_q, planner_output_q))
    memory_process = Process(target=text_agent_worker, args=(MEMORY_MODEL_PATH, "cuda:6", memory_input_q, memory_output_q))
    action_process = Process(target=multimodal_agent_worker, args=(ACTION_MODEL_PATH, "cuda:7", action_input_q, action_output_q))

    planner_process.start()
    action_process.start()
    memory_process.start()

    with open(RANDOM_SPLIT_FILE, 'r') as f:
        split_data = json.load(f)
    episodes_to_run = [f.split('.')[0] for f in split_data['test']]

    results = evaluate_trajectory_multigpu(episodes_to_run, planner_input_q, planner_output_q, action_input_q, action_output_q, memory_input_q, memory_output_q)

    planner_input_q.put((None, None, None)); action_input_q.put((None, None, None)); memory_input_q.put((None, None))
    planner_process.join(); action_process.join(); memory_process.join()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_filepath = os.path.join(OUTPUT_DIR, output_filename)
    
    with open(output_filepath, "w", encoding='utf-8') as f:
        for result_item in results:
            f.write(json.dumps(result_item) + "\n")
            