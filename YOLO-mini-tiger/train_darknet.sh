#! /bin/bash

S3_BASE=s3://calvinandpogs-ee148/atrw/detection/
DN_BASE=darknet/data/tiger/VOCdevkit/VOC2007/

# Clone Darknet
git clone https://github.com/AlexeyAB/darknet
rm darknet/Makefile
cp darknet_files/* darknet/

# Copy training data
mkdir $(DN_BASE)
aws s3 cp $(S3_BASE)/annotations/Annotations/ $(DN_BASE)Annotation/
aws s3 cp $(S3_BASE)/annotations/ImageSets/ $(DN_BASE)ImageSets/
aws s3 cp $(S3_BASE)train/ $(DN_BASE)JPEGImages/

# Build Darknet
cd darknet
make -j

# Train
./darknet detector train cfg/tiger.data cfg/yolo-mini-tiger.cfg
