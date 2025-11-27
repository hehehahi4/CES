#!/bin/bash
set -x
set -euo pipefail

project_dir="$(pwd)"

# TODO: Change your path
base_coordinator_model=/data2/home/dengzehao/data/model/planner-vl2-rl
base_executor_model=/data2/home/dengzehao/data/model/GUI-R1/GUI-R1-7B
base_manager_model=/data2/home/dengzehao/data/model/tracker-4B
manager_save_path=/data2/home/dengzehao/data/model/tracker-4B-rl
trainer_name=tracker-4B-train2


data_file=$project_dir/data/mydata/train_stage2/ces_data4.parquet
tool_config_path=$project_dir/examples/swirl/tool_config/gui_executor_tool_config.yaml
executor_config_path=$project_dir/examples/swirl/tool_config/executor_agent_config.json
coordinator_config_path=$project_dir/examples/swirl/tool_config/coordinator_agent_config.json


coordinator_server_ports=(10032 10033)
executor_server_ports=(10030 10031)
coordinator_ports_csv=$(IFS=,; echo "${coordinator_server_ports[*]}")
executor_ports_csv=$(IFS=,; echo "${executor_server_ports[*]}")


log_path=log/manager
mkdir -p $log_path

ray_tmpdir=$project_dir/ray
export ray_temp_dir="$ray_tmpdir"
export ray_tmpdir="$ray_tmpdir"


pkill -f "vllm serve" || true
sleep 10

# start coordinator and executor service
bash $project_dir/examples/swirl/start_vllm.sh $base_executor_model $executor_ports_csv $log_path/executor 0 &
bash $project_dir/examples/swirl/start_vllm.sh $base_coordinator_model $coordinator_ports_csv $log_path/coordinator 2 &

sleep 60

export no_proxy="localhost,127.0.0.1,0.0.0.0,10.5.2.61"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,10.5.2.61"


# train tracker
export CUDA_VISIBLE_DEVICES=4,5,6,7
python3 -m verl.trainer.main_ppo \
    --config-path="$project_dir/examples/swirl/running_config" \
    --config-name='ppo_trainer' \
    data.train_files=$data_file \
    data.val_files=$data_file \
    data.image_key=none \
    data.train_batch_size=32 \
    data.max_prompt_length=8192 \
    data.max_response_length=256 \
    actor_rollout_ref.model.path=$base_manager_model \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    +online_reweighting.lb=-0.1 \
    +online_reweighting.ub=2.0 \
    reward_model.reward_manager=tracker \
    reward_model.launch_reward_fn_async=true \
    +reward_model.reward_kwargs.coordinator_agent_config=$coordinator_config_path \
    +reward_model.reward_kwargs.executor_agent_config=$executor_config_path \
    +reward_model.reward_kwargs.tool_config_path=$tool_config_path \
    custom_reward_function.path=$project_dir/verl/utils/reward_score/gui.py \
    trainer.project_name=tracker \
    trainer.experiment_name=$trainer_name \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.total_epochs=5

sleep 30


python -m multi_agent.smart_model_merger \
    --local_dir checkpoints/tracker/$trainer_name \
    --target_dir $manager_save_path


pkill -f "vllm serve" || true

echo "manager training completed."