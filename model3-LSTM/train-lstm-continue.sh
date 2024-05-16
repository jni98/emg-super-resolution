#!/bin/bash -l

# Usage: sbatch train-lstm.sh [optional_checkpoint_path] [optional_results_dir]
# Example to start first job: sbatch train-lstm.sh
# Example to start second job manually: sbatch train-lstm.sh $HOME/model3-LSTM-<JOBID>/last.ckpt $HOME/model3-LSTM-<JOBID>

############ slurm ###############

#SBATCH --job-name=model3-LSTM
#SBATCH --time=24:00:00
#SBATCH --mail-user=janina.kloos@fau.de --mail-type=ALL
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=a100
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

############ paths ###############

SRC_DIR=$HOME/code/model3-LSTM
RESULTS_DIR=${2:-$HOME/model3-LSTM-$SLURM_JOB_ID}  # Use passed results directory or default to new one based on job ID
DATA_ARCHIVE=$HPCVAULT/dataset_lstm.tar
FAST_DATA_DIR=$TMPDIR/model3-LSTM-$SLURM_JOB_ID

############ params ###############

EPOCHS=50000
BATCHSIZE=2
LR=0.00001
CHECKPOINT_PATH=$1  # First parameter: optional checkpoint path

############ setup ###############

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


srun python trainer.py --data "$FAST_DATA_DIR/dataset_lstm/train" \
                  --testdata "$FAST_DATA_DIR/dataset_lstm/test" \
                  --results "$RESULTS_DIR" \
                  --epochs $EPOCHS --batch_size $BATCHSIZE --lr $LR --checkpoint "$CHECKPOINT_PATH"

rm -rf "$FAST_DATA_DIR"