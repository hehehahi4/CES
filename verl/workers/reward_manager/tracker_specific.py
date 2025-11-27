from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import torch
import json
from openai import OpenAI
from verl import DataProto
from verl.workers.reward_manager import register
from functools import partial

import numpy.random as npr
from verl.tools.utils.tool_registry import initialize_tools_from_config
from sglang.srt.openai_api.protocol import Tool

try:
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
except ImportError:
    from sglang.srt.function_call_parser import FunctionCallParser


def initialize_tools(tools_config_file):
    if tools_config_file is None:
        return [], {}, None, [], None

    tool_list = initialize_tools_from_config(tools_config_file)

    tool_schemas = [tool.get_openai_tool_schema().model_dump() for tool in tool_list]
    tool_call_parser_type = 'qwen25'
    sgl_tools = [Tool.model_validate(tool_schema) for tool_schema in tool_schemas]
    function_call_parser = FunctionCallParser(
        sgl_tools,
        tool_call_parser_type,
    )

    return function_call_parser


@register("tracker")
class TrackerRewardManager:

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", 
                 tool_config_path="", coordinator_agent_config="", executor_agent_config="") -> None:
        """
        Initialize the TrackerRewardManager instance for CES framework.
        
        It maintains connections to TWO external agents:
        1. Frozen Coordinator
        2. Frozen Executor
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key

        self.coord_creaters = []
        if coordinator_agent_config:
            coord_config = json.load(open(coordinator_agent_config))
            coord_urls = coord_config['coordinator_agent_urls']
            coord_model = coord_config['coordinator_agent_path']
            
            for url in coord_urls:
                client = OpenAI(base_url=url, api_key="None", timeout=10.0) 
                msg_create = partial(
                    client.chat.completions.create,
                    model=coord_model,
                )
                self.coord_creaters.append(msg_create)
        
        self.exec_creaters = []
        if executor_agent_config:
            exec_config = json.load(open(executor_agent_config))
            exec_urls = exec_config['executor_agent_urls']
            exec_model = exec_config['executor_agent_path']
            
            for url in exec_urls:
                client = OpenAI(base_url=url, api_key="None", timeout=30.0)
                msg_create = partial(
                    client.chat.completions.create,
                    model=exec_model,
                )
                self.exec_creaters.append(msg_create)

        self.coord_client_num = len(self.coord_creaters)
        self.exec_client_num = len(self.exec_creaters)
        
        if self.coord_client_num == 0 or self.exec_client_num == 0:
            print("Warning: Coordinator or Executor clients not initialized properly.")

        self.function_call_parser = initialize_tools(tool_config_path)


    def __call__(self, data: DataProto, return_dict=False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        jobs = []
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            # print(f"Prompt: {prompt_str}")

            state_response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            # print(f"State Tracker Response: {state_response_str}")

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key] 
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            if self.coord_client_num > 0:
                coord_idx = i % self.coord_client_num
                coord_client = self.coord_creaters[coord_idx]
            else:
                coord_client = None
                
            if self.exec_client_num > 0:
                exec_idx = i % self.exec_client_num
                exec_client = self.exec_creaters[exec_idx]
            else:
                exec_client = None

            jobs.append((i, valid_response_length, state_response_str, ground_truth, extra_info, coord_client, exec_client, data_source))

        def _compute_one(job):
            i, valid_response_length, state_str, ground_truth, extra_info, coord_client, exec_client, data_source = job
            
            score = self.compute_score(
                data_source=data_source,
                solution_str=state_str,        
                ground_truth=ground_truth,    
                extra_info=extra_info,
                function_call_parser=self.function_call_parser,
                coord_creater=coord_client,    
                exec_creater=exec_client  
            )
            return i, valid_response_length, score

        with ThreadPoolExecutor(max_workers=min(64, len(jobs))) as exe:
            futures = [exe.submit(_compute_one, job) for job in jobs]
            for future in as_completed(futures):
                i, valid_response_length, score = future.result()

                if isinstance(score, dict):
                    reward = score["score"]
                    for k, v in score.items():
                        reward_extra_info[k].append(v)
                else:
                    reward = score

                reward_tensor[i, valid_response_length - 1] = reward

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor