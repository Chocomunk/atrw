#!bin/bash

mkdir -p data/images/
mkdir -p data/labels/

mkdir -p ../coco/annotations/
cp data/atrw_detect.json ../coco/annotations/instances_val2017.json

echo "------------------------------SETUP ATRW-----------------------------------"

# Setup training data
if [[ -z "${SM_CHANNEL_TRAIN}" ]]; then    # Default to empty files
    touch train.txt
    touch val.txt
else
    cp "$SM_CHANNEL_ANNOT"/train.txt .
    cp "$SM_CHANNEL_ANNOT"/val.txt .
    
    # Generate subset file if specified
    if [[ $# -eq 2 ]]; then
        python data/scripts/generate_subsets.py --input train.txt --output "$1" --frac "$2"
    elif [[ $# -gt 0 ]]; then
        echo "setup_tiger.sh: Incorrect number of arguments $#. Expected 0 or 2\n\tArgs: {subset_file} {subset_proportion}" >/dev/stderr
    fi
    
    ln -s "$SM_CHANNEL_TRAIN" data/images/tiger
    ln -s "$SM_CHANNEL_LABEL" data/labels/tiger
fi

# Setup test data
if [[ -z "${SM_CHANNEL_TEST}" ]]; then    # Default to empty files
    touch test.txt
else
    cp "$SM_CHANNEL_ANNOT_TEST"/test.txt .
    
    ln -s "$SM_CHANNEL_TEST" data/images/tiger-test
    ln -s "$SM_CHANNEL_LABEL_TEST" data/labels/tiger-test
fi


# Finish background tasks
wait
