#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name audioldm_generate
#SBATCH --nodes=1
#SBATCH --time=10:00
#SBATCH --error=error_test
#SBATCH --output=output_test
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=2

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURM_NTASKS="$SLURM_NTASKS
ulimit -s unlimited
ulimit -c unlimited

python3 audioldm_train/infer.py --config_yaml audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_custom.yaml --list_inference tests/captionlist/inference_test.lst --reload_from_ckpt log/latent_diffusion/2023_08_23_reproduce_audioldm/audioldm_custom/checkpoints/checkpoint-fad-133.00-global_step=69999.ckpt
