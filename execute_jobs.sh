#!/bin/bash

# Set default values for optional arguments
LIVE_ANCHOR=${1:-false}
TEST_MODE=${2:-true}

# Check if required arguments are provided
if [ -z "$3" ]; then
  echo "Usage: $0 <live_anchor> <test_mode> <input_file>"
  exit 1
fi

input_file="$3"

# Loop over each Reuters ID in the input file
while IFS= read -r REUTERS_ID
do
    echo "Executing job for REUTERS_ID: $REUTERS_ID"
    
    # Execute the gcloud command for each REUTERS_ID
    gcloud run jobs execute video-job \
        --region=us-central1 \
        --update-env-vars ^@^LIVE_ANCHOR=$LIVE_ANCHOR@TEST_MODE=$TEST_MODE@REUTERS_ID=$REUTERS_ID \
        --async
    
    echo "Job execution completed for REUTERS_ID: $REUTERS_ID"
    echo "---"
done < "$input_file"
