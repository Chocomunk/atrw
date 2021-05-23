#!bin/bash

mkdir -p data/images/
mkdir -p data/labels/

echo "------------------------------DOWNLOAD ATRW-----------------------------------"

# Copy label ids
cp "$SM_CHANNEL_ANNOTS"/train.txt .
cp "$SM_CHANNEL_ANNOTS"/val.txt .

# Link data to proper directories
ln -s "$SM_CHANNEL_IMAGES" data/images/tiger
ln -s "$SM_CHANNEL_LABELS" data/labels/tiger

# Finish background tasks
wait
