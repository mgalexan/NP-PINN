#!/bin/bash
#SBATCH --mail-user=mgalexan@uwaterloo.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=NP-PINN
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=05:00:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=2
#SBATCH --output=./GPU_job.out
#SBATCH --error=./GPU_job.err

# Activate Conda
module load slurm
source /work/mgalexan/miniconda3/etc/profile.d/conda.sh
conda activate working-env

# Move to project
cd /work/mgalexan/NP-PINN

# Check environment
echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi
which python
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

# Run your script
python test.py
