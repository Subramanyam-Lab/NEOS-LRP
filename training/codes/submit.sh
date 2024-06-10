#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128GB
#SBATCH --job-name=name
#SBATCH --output=output.out
#SBATCH --time=7-00:00:00
#SBATCH --error=error.err
#SBATCH --account=azs7266_p_gpu
#SBATCH --partition=sla-prio
#SBATCH --gpus=1

source ~/.bashrc
conda activate /path/to/conda/env

declare -a num_instances_array=("100" "1000" "10000" "100000" "1000000")

for num_instances in "${num_instances_array[@]}"
do
    export num_instances="$num_instances"

    echo "Running  with num_instances=$num_instances"

    python problem.py --surrogate_model GP
done

