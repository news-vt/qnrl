#!/bin/bash

## Get information for current job.

echo "Job specs:"
echo "   SLURM_JOB_ID: $SLURM_JOB_ID"
echo "   DATE: $(date)"
echo "   TIME_LIMIT: $(squeue -j $SLURM_JOB_ID -h --Format TimeLimit)"
echo "   TIME_ELAPSED: $(squeue -j $SLURM_JOB_ID -h -o "%M")"
echo "   SLURM_NNODES: $SLURM_NNODES"
echo "   SLURM_NTASKS: $SLURM_NTASKS"
echo "   SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"