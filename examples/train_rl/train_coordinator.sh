#!/bin/bash
set -x
set -euo pipefail

project_dir="$(pwd)"

# TODO: Change your path
base_interactor_model=/data2/home/dengzehao/data/model/GUI-R1/GUI-R1-7B
base_coordinator_model=/data2/home/dengzehao/data/model/planner-vl2
coordinator_save_path=/data2/home/dengzehao/data/model/planner-vl2-rl
trainer_name=planner

data_file=$project_dir/data/mydata/train_stage1/ces_data4.parquet
tool_config_path=$project_dir/examples/swirl/tool_config/gui_executor_tool_config.yaml
grounding_config_path=$project_dir/examples/swirl/tool_config/executor_agent_config.json


grounding_server_ports=(10030 10031 10032 10033)
ports_csv=$(IFS=,; echo "${grounding_server_ports[*]}")


log_path=log/coordinator
mkdir -p $log_path

ray_tmpdir=$project_dir/ray
export ray_temp_dir="$ray_tmpdir"
export ray_tmpdir="$ray_tmpdir"

pkill -f "vllm serve" || true
sleep 10

# start interactor service
bash $project_dir/examples/swirl/start_vllm.sh $base_interactor_model $ports_csv $log_path &
sleep 60


# train coordinator
export CUDA_VISIBLE_DEVICES=4,5,6,7
python3 -m verl.trainer.main_ppo \
    --config-path="$project_dir/examples/swirl/running_config" \
    --config-name='ppo_trainer' \
    data.train_files=$data_file \
    data.val_files=$data_file \
    data.train_batch_size=32 \
    data.max_prompt_length=8192  \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=$base_coordinator_model \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    +online_reweighting.lb=-0.1 \
    +online_reweighting.ub=2.0 \
    reward_model.reward_manager=planner \
    reward_model.launch_reward_fn_async=true \
    +reward_model.reward_kwargs.executor_agent_config=$grounding_config_path \
    +reward_model.reward_kwargs.tool_config_path=$tool_config_path \
    custom_reward_function.path=$project_dir/verl/utils/reward_score/gui.py \
    trainer.project_name=coordinator \
    trainer.experiment_name=$trainer_name \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.total_epochs=10

sleep 30

python -m multi_agent.smart_model_merger \
    --local_dir checkpoints/coordinator/$trainer_name \
    --target_dir $coordinator_save_path


pkill -f "vllm serve" || true

echo "training completed."