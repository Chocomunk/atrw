#! /bin/bash

S3_BASE=s3://calvinandpogs-ee148/atrw/out/pose/hrnet/$(date +%m-%d-%y-%H-%M-%S)
HRN_BASE=deep-high-resolution-net.pytorch

# Clone HRNet
# 1
#pip install torch >= 1.0.0
#2
git clone https://github.com/wanghao14/CVWC2019-pose
#3
cd CVWC2019-pose/deep-high-resolution-net.pytorch
pip install -r requirements.txt
#4
cd lib
make
#5
cd ../../cocoapi/PythonAPI
make install
python3 setup.py install --user
#6
cd ../../deep-high-resolution-net.pytorch
mkdir output 
mkdir log
#7 downloaded manually
#pip install gdown
#gdown --id 1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC

cd ../

# Copy training data
mkdir -p "$HRN_BASE"

ln -s "$SM_CHANNEL_POSE" "$HRN_BASE"/data/atrw/pose
ln -s "$SM_CHANNEL_MODELS" "$HRN_BASE"/models

# Train
cd deep-high-resolution-net.pytorch
python tools/train.py --cfg expriments/atrw/w48_384x288.yaml -dont_show 2>&1 | tee train_output.txt

# Test
python tools/test.py --cfg expriments/atrw/w48_384x288.yaml 2>&1 | tee test_output.txt

# Copy data back
aws s3 cp output "$S3_BASE"/output
aws s3 cp log "$S3_BASE"/log
aws s3 cp train_output.txt "$S3_BASE"/
aws s3 cp test_output.txt "$S3_BASE"/