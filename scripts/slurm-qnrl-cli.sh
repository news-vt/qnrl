#!/bin/bash
## Reference: https://www.docs.arc.vt.edu/resources/compute/01owl.html
## Usage:
## $ sbatch ./scripts/slurm-qnrl-cli.sh --config ./experiments/qnrl_CartPole-v1_quantum-QuantumNativeQNetworkSplit-partition.py train

## You will need your own account.
#SBATCH --account="bigdata"

# Partition name. Options change based on cluster.
#SBATCH --partition=normal_q

## Specifying that you must have this job run on the
## Owl AMD EPYC 9454 Zen4 "Genoa" nodes.
#SBATCH --constraint=genoa

## This next line is optional because it is the default.
#SBATCH --qos=owl_normal_base

## Time limit for the job. The job will be cancelled if the runtime exceeds this.
#SBATCH --time=48:00:00

## Compute resources.
### Number of compute nodes.
#SBATCH --nodes=1
### Number of tasks can be thought of as number of processes
### as in the case of MPI.
#SBATCH --ntasks-per-node=1
### Number of cpus/cores is the number of cores needed
### to run each task (e.g., for parallelism).
#SBATCH --cpus-per-task=24

# Reset modules.
module reset

# Load particular modules.
module load Miniconda3

# Initialize the shell to use Anaconda.
eval "$(conda shell.bash hook)"

# Source any virtual environments.
conda activate qnrl

echo "Job $SLURM_JOB_ID has started on node"
echo "Job args: $@"
echo "Job specs:"
echo "   SLURM_JOB_ID: $SLURM_JOB_ID"
echo "   DATE: $(date)"
echo "   TIME_LIMIT: $(squeue -j $SLURM_JOB_ID -h --Format TimeLimit)"
echo "   TIME_ELAPSED: $(squeue -j $SLURM_JOB_ID -h -o "%M")"
echo "   SLURM_NNODES: $SLURM_NNODES"
echo "   SLURM_NTASKS: $SLURM_NTASKS"
echo "   SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"

# Run the job.
python -m qnrl.cli $@


# ## -----------------------
# ## EXPORTS
# ## Exports and variable assignments.
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export MV2_ENABLE_AFFINITY=0 

# echo "   SLURM_CPUS_PER_TASK: " $SLURM_CPUS_PER_TASK
# echo "   OMP_NUM_THREADS: " $OMP_NUM_THREADS
# echo "   MV2_ENABLE_AFFINITY: "  $MV2_ENABLE_AFFINITY
# echo "   SLURM_NTASKS: "  $SLURM_NTASKS 
 

# ## -----------------------
# ## MPI JOB.
# THE_INPUT="./nim-11-1-1.inp"
# THE_EXEC="../bin-owl-openmpi/is-hybrid-owl-openmpi"
# srun --mpi=pmix  $THE_EXEC  $THE_INPUT