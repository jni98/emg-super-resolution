#!/bin/bash

# Submit the first job
JOBID1=$(sbatch.tinygpu train-model2.sh | awk '{print $4}')

# Submit the second job with a dependency to start after the first job completes under any circumstance
sbatch.tinygpu --dependency=afterany:$JOBID1 train-model2-continue.sh $HOME/model2-13x5-Layer-interpolated-Lightning-$JOBID1/last.ckpt
