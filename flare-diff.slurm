#!/bin/bash
#SBATCH --job-name=flare_train
#SBATCH --partition=gpu             # Replace with your cluster's actual GPU partition name
#SBATCH --nodes=1                   # Request 1 node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=8           # Number of CPU cores requested
#SBATCH --mem=48G                   # Request 48 GB of RAM
#SBATCH --gres=gpu:nvidia:1                # Request 1 NVIDIA GPU
#SBATCH --time=48:00:00             # Max wall time (Hours:Minutes:Seconds)
#SBATCH --output=/home/nate/flare-diff-out/slurm-%j.out # Standard output and error log

# 1. Create the specific output directory for this job's checkpoints and images
OUT_DIR="/home/nate/flare-diff-out/${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

echo "Starting job $SLURM_JOB_ID on $(hostname)"
echo "Output directory created at: $OUT_DIR"

# 2. Load your environment (Uncomment and adjust these lines based on your cluster)
# source /home/software/miniforge3/etc/profile.d/conda.sh
# conda activate acd

# 3. Navigate to the directory containing your code
# cd /home/nate/ACD_SEP/codebase

# 4. Run the training script
# We pass the datadir, the dynamic save_folder, and the suffix to trigger the SWANSF logic.
python train.py \
    --datadir /scratch/nate/SWANSF-diff \
    --save_folder "$OUT_DIR" \
    --suffix flare_diff \
    --epochs 750 \
    --batch_size 32 
    --encoder mlp
    --decoder mlp