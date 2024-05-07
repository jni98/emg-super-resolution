#!/bin/bash -l

# Usage: sbatch train-lstm.sh [optional_checkpoint_path]
# Example to start first job: sbatch train-lstm.sh
# Example to start second job manually (if needed): sbatch train-lstm.sh $HOME/model3-LSTM-<JOBID>/best_model.ckpt

############    slurm    ###############

#SBATCH --job-name=model3-LSTM
#SBATCH --time=1:00:00
#SBATCH --mail-user=janina.kloos@fau.de --mail-type=ALL
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

############    paths    ###############

SRC_DIR=$HOME/code/model3-LSTM
RESULTS_DIR=$HOME/model3-LSTM-$SLURM_JOB_ID
DATA_ARCHIVE=$HPCVAULT/dataset_lstm.tar
FAST_DATA_DIR=$TMPDIR/model3-LSTM-$SLURM_JOB_ID

############    params   ###############

EPOCHS=5
BATCHSIZE=4
LR=0.00001

############    setup    ###############

# allows for internet connection on cluster nodes
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

# load and init conda if it doesn't exist
if ! command -v conda &> /dev/null
then
    echo "load conda from module"
    module load python/3.10-anaconda
    conda init bash
fi

# activate conda env
source activate thesis_env
echo "using python $(which python)"

# check that python can access GPU
nvidia-smi
echo "python reaches gpu: $(python -c 'import torch;print(torch.cuda.is_available())')"

# copy training data to faster drive
echo "started data transfer at $(date)"
mkdir "$FAST_DATA_DIR"
tar -xf "$DATA_ARCHIVE" -C "$FAST_DATA_DIR"
echo "finished transfer at $(date)"
cd "$FAST_DATA_DIR" || echo "could not cd into $FAST_DATA_DIR"
du -ah -d 1

# change to src
cd "$SRC_DIR" || echo "could not cd into $SRC_DIR"
echo "Using Repository Revision"
git log --oneline -n 1

############    training   ###############

CHECKPOINT_PATH=$1  # Optionally pass a checkpoint path when calling this script

echo "CHECKPOINT_PATH at start: $CHECKPOINT_PATH"

# Command to start training, possibly including logic based on $CHECKPOINT_PATH's state
python trainer.py --data "$FAST_DATA_DIR"/dataset_lstm/train --testdata "$FAST_DATA_DIR"/dataset_lstm/test --results "$RESULTS_DIR" --epochs $EPOCHS --batch_size $BATCHSIZE --lr $LR --checkpoint "$CHECKPOINT_PATH"

# Clean up operations
rm -rf "$FAST_DATA_DIR"

echo "CHECKPOINT_PATH before scheduling next job: $CHECKPOINT_PATH"

# Check if the script was called without a checkpoint path and schedule the next job
if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Scheduling next job without initial checkpoint."
    sbatch --dependency=afterok:$SLURM_JOB_ID $0 $RESULTS_DIR/best_model.ckpt
else
    echo "Checkpoint was provided; no new job scheduled."
fi
