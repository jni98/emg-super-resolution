#!/bin/bash

# Submit the first job and capture its JOBID
JOBID1=$(sbatch.tinygpu train-lstm.sh | awk '{print $4}')
echo "First job submitted with JOBID $JOBID1"

# Check if the first job submission was successful
if [ -z "$JOBID1" ]
then
    echo "Failed to submit first job. Exiting script."
    exit 1
fi

# Set the results directory using the first job's JOBID
RESULTS_DIR1=$HOME/model3-LSTM-$JOBID1

# Submit the second job with a dependency to start after the first job completes under any circumstance
JOBID2=$(sbatch.tinygpu --dependency=afterany:$JOBID1 train-lstm.sh $RESULTS_DIR1/last.ckpt $RESULTS_DIR1 | awk '{print $4}')
echo "Second job submitted with JOBID $JOBID2 depending on JOBID $JOBID1"

# Ensure that the second job submission is acknowledged before submitting the third job
if [ -z "$JOBID2" ]
then
    echo "Failed to submit second job. Exiting script."
    exit 1
else
    # Submit the third job with a dependency to start after the second job completes under any circumstance
    JOBID3=$(sbatch.tinygpu --dependency=afterany:$JOBID2 train-lstm.sh $RESULTS_DIR1/last.ckpt $RESULTS_DIR1 | awk '{print $4}')
    echo "Third job submitted with JOBID $JOBID3 depending on JOBID $JOBID2"
fi

# Ensure that the second job submission is acknowledged before submitting the third job
if [ -z "$JOBID3" ]
then
    echo "Failed to submit second job. Exiting script."
    exit 1
else
    # Submit the third job with a dependency to start after the second job completes under any circumstance
    JOBID4=$(sbatch.tinygpu --dependency=afterany:$JOBID3 train-lstm.sh $RESULTS_DIR1/last.ckpt $RESULTS_DIR1 | awk '{print $4}')
    echo "Job submitted with JOBID $JOBID4 depending on JOBID $JOBID3"
fi

# Ensure that the second job submission is acknowledged before submitting the third job
if [ -z "$JOBID4" ]
then
    echo "Failed to submit second job. Exiting script."
    exit 1
else
    # Submit the third job with a dependency to start after the second job completes under any circumstance
    JOBID5=$(sbatch.tinygpu --dependency=afterany:$JOBID4 train-lstm.sh $RESULTS_DIR1/last.ckpt $RESULTS_DIR1 | awk '{print $4}')
    echo "Job submitted with JOBID $JOBID5 depending on JOBID $JOBID4"
fi
