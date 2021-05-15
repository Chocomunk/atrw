#! /bin/bash

S3_BASE=s3://calvinandpogs-ee148/atrw/detection
DN_BASE=darknet/data/tiger/VOCdevkit/VOC2007

# Clone Darknet
git clone https://github.com/AlexeyAB/darknet
rm darknet/Makefile
cp -r darknet_files/* darknet/

# Copy training data
mkdir -p "$DN_BASE"

# aws s3 cp --recursive "$S3_BASE"annotations/Annotations/ "$DN_BASE"Annotation/
# aws s3 cp --recursive "$S3_BASE"annotations/ImageSets/ "$DN_BASE"ImageSets/
# aws s3 cp --recursive "$S3_BASE"train/ "$DN_BASE"JPEGImages/

ln -s "$SM_CHANNEL_ANNOT"/Annotations "$DN_BASE"/Annotations
ln -s "$SM_CHANNEL_ANNOT"/ImageSets "$DN_BASE"/ImageSets
ln -s "$SM_CHANNEL_TRAIN" "$DN_BASE"/JPEGImages

# Build Darknet
cd darknet
make -j"$SM_NUM_CPUS" >/dev/null
cd data/tiger
python voc_label.py
cd ../..

# Train
./darknet detector train cfg/tiger.data cfg/yolo-mini-tiger.cfg -dont_show

# Evaluate
./darknet detector map cfg/tiger.data cfg/yolo-mini-tiger.cfg backup/yolo-mini-tiger_final.weights
./darknet detector valid cfg/tiger.data cfg/yolo-mini-tiger.cfg backup/yolo-mini-tiger_final.weights -out ""

# Copy data back
aws s3 cp backup/yolo-mini-tiger_final.weights "$S3_BASE"/out/yolo-mini-tiger.weights
