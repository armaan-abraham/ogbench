#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=iris
#SBATCH --account=iris
#SBATCH --output=/iris/u/armaana/jobs/logs/%A_%a.out
#SBATCH --error=/iris/u/armaana/jobs/logs/%A_%a.err
#SBATCH --time=1:30:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="24G|48G"
#SBATCH --array=0-2

COMMANDS=(
    "python main_actionless.py --run_group=antmaze-medium --env_name=antmaze-medium-navigate-v0 --agent=agents/crl_actionless.py --agent.alpha=0.1"
    "python main_actionless.py --run_group=antmaze-medium --env_name=antmaze-medium-navigate-v0 --agent=agents/gcivl_actionless.py --agent.alpha=10.0"
    "python main_actionless.py --run_group=antmaze-medium --env_name=antmaze-medium-navigate-v0 --agent=agents/vcrl.py --agent.velocity_encoding_dim=8 --agent.alpha=0.2 --agent.velocity_encoder_hidden_dims=[256,] --agent.action_velocity_map_hidden_dims=[256,]"
)

export MUJOCO_GL=egl
. /iris/u/armaana/ogbench/.venv/bin/activate
cd /iris/u/armaana/ogbench/impls

# Get the command for this array task
COMMAND=${COMMANDS[$SLURM_ARRAY_TASK_ID]}

# Run it
echo "Task $SLURM_ARRAY_TASK_ID: $COMMAND"
echo "Started at: $(date)"

eval $COMMAND

echo "Finished at: $(date)"