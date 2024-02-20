#!/bin/bash -l

## Usage:  sbatch train-script.sh

############    slurm    ###############

#SBATCH --job-name=model2-13x5
#SBATCH --time=2:00:00
#SBATCH --mail-user=janina.kloos@fau.de
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100

############    paths    ###############

SRC_DIR=$HOME/code/model2-13x5
RESULTS_DIR=$HOME/model2-13x5-$SLURM_JOB_ID
DATA_ARCHIVE=$HPCVAULT/dataset.tar
FAST_DATA_DIR=$TMPDIR/model2-13x5-$SLURM_JOB_ID

############    params   ###############

EPOCHS=50
BS=16
LR=0.0001

############    params   ###############

# allows for internet connection on cluster nodes
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

# load and init conda if it doesnt exist
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
echo "Contents of $FAST_DATA_DIR/dataset/train:"
ls "$FAST_DATA_DIR/dataset/train"
du -ah -d 1

# change to src
cd "$SRC_DIR" || echo "could not cd into $SRC_DIR"
echo "Using Repository Revision"
git log --oneline -n 1

# start training
python train.py --data "$FAST_DATA_DIR"/dataset/train --testdata "$FAST_DATA_DIR"/dataset/test --results "$RESULTS_DIR" --epochs $EPOCHS --bs $BS --lr $LR

# cleanup
rm -rf "$FAST_DATA_DIR"
