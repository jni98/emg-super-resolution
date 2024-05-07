#!/bin/bash

# Submit the first job
JOBID1=$(sbatch.tinygpu train-lstm.sh | awk '{print $4}')

# Submit the second job with a dependency to start after the first job completes under any circumstance
sbatch.tinygpu --dependency=afterany:$JOBID1 train-lstm-continue.sh $HOME/model3-LSTM-$JOBID1/best_model.ckpt
