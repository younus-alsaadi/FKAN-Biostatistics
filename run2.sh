#!/bin/bash -e 
#SBATCH --time=0-48:00:00
#SBATCH --output=slurm/output-CAN-%x-%j.log
#SBATCH --error=slurm/errors-CAN-%x-%j.log
#SBATCH --partition=topfgpu                     
#SBATCH --constraint="A100"
#SBATCH --open-mode append


# Load Micromamba environment
eval "$(micromamba shell hook --shell=bash)"
micromamba activate /gpfs/cssb/user/alsaadiy/micromamba/envs/tf/
# 1. Initialize the module system
source /etc/profile.d/modules.sh  

# 2. Load CUDA module
module load cuda/11.8

# 3. Verify TensorFlow GPU detection
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPUs detected:', tf.config.list_physical_devices('GPU'))"

# Set the project path
PROJECT_PATH="/gpfs/cssb/user/alsaadiy/FKAN-Biostatistics/src"
cd "$PROJECT_PATH"

# Run the training script
python training_cnn.py
