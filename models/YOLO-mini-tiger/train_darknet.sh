#! /bin/bash

S3_OUT=s3://calvinandpogs-ee148/atrw/out/detection/yolo-mini/$(date +%m-%d-%y-%H-%M-%S)
DN_BASE=darknet/data/tiger/VOCdevkit/VOC2007

# Clone Darknet
git clone https://github.com/AlexeyAB/darknet
rm darknet/Makefile
cp -r darknet_files/* darknet/

# Copy training data
mkdir -p "$DN_BASE"

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
./darknet detector train cfg/tiger.data cfg/yolo-mini-tiger.cfg -dont_show 2>&1 | tee train_output.txt

# Evaluate
./darknet detector map cfg/tiger.data cfg/yolo-mini-tiger.cfg backup/yolo-mini-tiger_final.weights 2>&1 | tee map_output.txt
./darknet detector valid cfg/tiger.data cfg/yolo-mini-tiger.cfg backup/yolo-mini-tiger_final.weights -out "" 2>&1 | tee valid_output.txt

# Copy data back
aws s3 cp backup/yolo-mini-tiger_final.weights "$S3_OUT"/yolo-mini-tiger.weights
aws s3 cp train_output.txt "$S3_OUT"/
aws s3 cp map_output.txt "$S3_OUT"/
aws s3 cp valid_output.txt "$S3_OUT"/
